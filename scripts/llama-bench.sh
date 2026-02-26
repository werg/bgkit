#!/usr/bin/env bash
# Benchmark llama-server to find optimal --parallel and --ctx-size settings.
#
# Sweeps configurations, measures throughput and memory, writes results to CSV.
#
# Usage: scripts/llama-bench.sh [OUTPUT_FILE]

set -euo pipefail

COMPOSE="docker compose -f docker/docker-compose.yaml"
OUTPUT="${1:-data/llama-bench-results.csv}"

# Representative prompts (similar to description pipeline workload)
PROMPT_SHORT='Describe what the following source file does in 1-3 sentences. File: utils.py\n\ndef add(a, b):\n    return a + b'
PROMPT_MEDIUM='Describe what the directory/module provides in 2-4 sentences. Focus on purpose and API.\n\nFiles:\n- auth.py: Handles user authentication via JWT tokens\n- middleware.py: Express middleware for route protection\n- utils.py: Password hashing and token generation helpers'

mkdir -p "$(dirname "$OUTPUT")"

# CSV header
if [[ ! -f "$OUTPUT" ]]; then
    echo "timestamp,model,parallel,ctx_size,ok,fired,total_time_s,avg_latency_s,p95_latency_s,throughput_req_per_s,gpu_memory_mb" > "$OUTPUT"
fi

wait_healthy() {
    local port="$1"
    local max_wait="${2:-120}"
    local elapsed=0
    while ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [[ $elapsed -ge $max_wait ]]; then
            echo "ERROR: Server on port $port not healthy after ${max_wait}s" >&2
            return 1
        fi
    done
}

warmup_server() {
    local port="$1"
    curl -sf -H 'Content-Type: application/json' \
        "http://localhost:${port}/v1/chat/completions" \
        -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"chat_template_kwargs":{"enable_thinking":false}}' > /dev/null 2>&1 || true
    sleep 1
}

fire_requests() {
    local port="$1"
    local n_requests="$2"
    local prompt="$3"
    local tmpdir
    tmpdir=$(mktemp -d)

    local start_time
    start_time=$(date +%s%N)

    # Fire concurrent requests
    for i in $(seq 1 "$n_requests"); do
        (
            local req_start req_end
            req_start=$(date +%s%N)
            if curl -sf -H 'Content-Type: application/json' \
                "http://localhost:${port}/v1/chat/completions" \
                -d "{\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":256,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
                -o /dev/null 2>/dev/null; then
                req_end=$(date +%s%N)
                echo "$(( (req_end - req_start) / 1000000 ))" > "${tmpdir}/${i}.ms"
            fi
        ) &
    done
    wait

    local end_time
    end_time=$(date +%s%N)
    local total_ms=$(( (end_time - start_time) / 1000000 ))

    # Collect latencies (only from successful requests)
    local latencies=()
    for f in "${tmpdir}"/*.ms; do
        [[ -f "$f" ]] && latencies+=("$(cat "$f")")
    done
    rm -rf "$tmpdir"

    local n=${#latencies[@]}
    if [[ $n -eq 0 ]]; then
        echo "${total_ms},0,0,0"
        return
    fi

    # Sort latencies for p95
    IFS=$'\n' sorted=($(sort -n <<<"${latencies[*]}")); unset IFS
    local p95_idx=$(( n * 95 / 100 ))
    [[ $p95_idx -ge $n ]] && p95_idx=$((n - 1))

    local sum=0
    for lat in "${sorted[@]}"; do
        sum=$((sum + lat))
    done
    local avg_ms=$((sum / n))

    echo "${total_ms},${avg_ms},${sorted[$p95_idx]},${n}"
}

get_gpu_memory() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0"
}

PORT="${LLAMA_PORT:-8080}"
MODEL="${LLAMA_MODEL:-Qwen3-0.6B-Q8_0.gguf}"

echo "Llama-server benchmark"
echo "Results: $OUTPUT"
echo ""

for parallel in 8 16 24 32 48; do
    for ctx in 32768 65536 131072 262144; do
        echo "=== parallel=${parallel} ctx_size=${ctx} ==="

        LLAMA_PARALLEL="$parallel" LLAMA_CTX="$ctx" \
            $COMPOSE up -d --force-recreate llama

        if ! wait_healthy "$PORT" 120; then
            echo "SKIP (unhealthy)"
            continue
        fi
        warmup_server "$PORT"

        # Fire N concurrent requests matching parallel slots
        result=$(fire_requests "$PORT" "$parallel" "$PROMPT_MEDIUM")
        IFS=',' read -r total_ms avg_ms p95_ms n_ok <<< "$result"

        gpu_mem=$(get_gpu_memory)

        total_s=$(awk "BEGIN{printf \"%.2f\", ${total_ms}/1000}")
        avg_s=$(awk "BEGIN{printf \"%.3f\", ${avg_ms}/1000}")
        p95_s=$(awk "BEGIN{printf \"%.3f\", ${p95_ms}/1000}")
        throughput=$(awk "BEGIN{printf \"%.2f\", ${n_ok}/(${total_ms}/1000)}")

        ts=$(date -Iseconds)
        echo "${ts},${MODEL},${parallel},${ctx},${n_ok},${parallel},${total_s},${avg_s},${p95_s},${throughput},${gpu_mem}" >> "$OUTPUT"
        echo "  ok=${n_ok}/${parallel} total=${total_s}s avg=${avg_s}s p95=${p95_s}s throughput=${throughput}req/s gpu=${gpu_mem}MB"
    done
done

echo ""
echo "=== Results ==="
column -t -s',' "$OUTPUT" 2>/dev/null || cat "$OUTPUT"
echo ""
echo "Full results: $OUTPUT"
