"""bgkit ↔ natstack supervisor bridge (additive, env-gated).

When ``BGKIT_BRIDGE_SOCK`` is set, the trainer connects to that unix domain
socket (created by the natstack shell extension before spawning us) and speaks
the bidirectional NDJSON bridge protocol defined in
``workspace/workers/bgkit-supervisor/frames.ts``. It EMITS telemetry
(hello / metric / status / event / knobs / ack) and RECEIVES control commands
(set_hp / request_eval / save_checkpoint / pause / resume / list_knobs).

This is a faithful mirror of the reference implementation in
``tools/bgkit-stub-train.py`` — identical frames, identical camelCase keys.

Design constraints (deliberate):
  * Import is stdlib-only and lazy — importing this module never pulls torch,
    so it is safe to import unconditionally from the trainer.
  * When ``BGKIT_BRIDGE_SOCK`` is unset, ``maybe_create()`` returns ``None`` and
    the trainer leaves every hook guarded by ``if bridge is not None:`` — so the
    env-unset path is byte-for-byte unchanged.
  * Inbound control frames are buffered on a thread-safe queue by a background
    reader thread, but APPLIED only when the trainer calls ``drain_control()``
    at its own step boundary — so live control lands at the SAME point as the
    existing file-based ``LiveConfig.poll()`` and never mutates trainer state
    from a foreign thread.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import sys
import threading
import time

# camelCase wire keys to match the TypeScript contract verbatim — no mapping on
# the consumer hot path.
_CONNECT_ATTEMPTS = 50
_CONNECT_BACKOFF_S = 0.1
_RECV_CHUNK = 65536


class NatstackBridge:
    """Owns the bridge socket, locked outbound writes, applied_seq accounting,
    an inbound control queue, and pause/eval/checkpoint flags.

    Construct via :meth:`maybe_create` (returns ``None`` when the env var is
    unset) rather than directly, so call-sites stay one guarded line.
    """

    def __init__(self, sock_path: str) -> None:
        self._sock_path = sock_path
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._closed = False

        # Inbound control buffer — populated by the reader thread, drained by
        # the trainer at its step boundary.
        self._control_q: queue.Queue[dict] = queue.Queue()

        # Highest control seq the trainer has applied (HP-change ack). Surfaced
        # in status frames as ``appliedSeq``.
        self.applied_seq = 0

        # Loop-consultable flags. The trainer reads/clears these at the step
        # boundary; they are set from the drained control frames, never from the
        # reader thread.
        self.paused = False
        self.eval_requested = False
        self.checkpoint_requested = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def maybe_create(cls) -> "NatstackBridge | None":
        """Return a connected bridge if ``BGKIT_BRIDGE_SOCK`` is set, else None.

        Connection failures degrade to ``None`` rather than raising, so a
        misconfigured supervisor socket can never crash training.
        """
        sock_path = os.environ.get("BGKIT_BRIDGE_SOCK")
        if not sock_path:
            return None
        bridge = cls(sock_path)
        if not bridge.connect():
            return None
        return bridge

    def connect(self) -> bool:
        """Connect to the bridge socket (with retry) and start the reader
        thread. Returns True on success.
        """
        # The extension creates the socket before spawning us, but tolerate a
        # startup race the same way the reference stub does.
        for _ in range(_CONNECT_ATTEMPTS):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self._sock_path)
                self._sock = s
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(_CONNECT_BACKOFF_S)
            except OSError:
                break
        if self._sock is None:
            print(
                f"[natstack_bridge] could not connect to bridge "
                f"{self._sock_path}; bridge disabled",
                file=sys.stderr,
            )
            return False
        self._reader = threading.Thread(
            target=self._read_loop, name="natstack-bridge-reader", daemon=True
        )
        self._reader.start()
        return True

    def close(self) -> None:
        self._closed = True
        with self._lock:
            s, self._sock = self._sock, None
        if s is not None:
            try:
                s.close()
            except OSError:
                pass

    # ── outbound (telemetry) ─────────────────────────────────────────────────
    def _send(self, frame: dict) -> None:
        line = (json.dumps(frame, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            if self._sock is None:
                return
            try:
                self._sock.sendall(line)
            except OSError:
                # Supervisor went away — drop the socket and degrade silently.
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def emit_hello(
        self, phase: str, config: dict, max_steps: int | None = None
    ) -> None:
        frame: dict = {
            "t": "hello",
            "pid": os.getpid(),
            "phase": phase,
            "config": config,
        }
        if max_steps is not None:
            frame["maxSteps"] = int(max_steps)
        self._send(frame)

    def emit_metric(self, step: int, source: str, data: dict) -> None:
        """Emit a per-step metric bundle. ``source`` is "train" or "eval".

        Non-numeric values are dropped so the frame matches the wire contract
        (``Record<string, number>``).
        """
        clean: dict = {}
        for k, v in data.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                clean[k] = v
        self._send(
            {
                "t": "metric",
                "step": int(step),
                "wallMs": int(time.monotonic() * 1000),
                "source": source,
                "data": clean,
            }
        )

    def emit_status(
        self,
        step: int,
        hps: dict,
        applied_seq: int | None = None,
        phase: str | None = None,
        max_steps: int | None = None,
    ) -> None:
        """Emit the ground-truth applied-config snapshot.

        ``hps`` carries only number/bool/str values per the contract.
        """
        clean: dict = {
            k: v
            for k, v in hps.items()
            if isinstance(v, (int, float, bool, str))
        }
        frame: dict = {
            "t": "status",
            "step": int(step),
            "hps": clean,
            "appliedSeq": int(
                self.applied_seq if applied_seq is None else applied_seq
            ),
        }
        if phase is not None:
            frame["phase"] = phase
        if max_steps is not None:
            frame["maxSteps"] = int(max_steps)
        self._send(frame)

    def emit_event(self, step: int, kind: str, data: dict | None = None) -> None:
        frame: dict = {"t": "event", "step": int(step), "kind": kind}
        if data:
            frame["data"] = data
        self._send(frame)

    def emit_knobs(self, schema: list[dict], reply_to: int | None = None) -> None:
        frame: dict = {"t": "knobs", "schema": schema}
        if reply_to is not None:
            frame["replyTo"] = int(reply_to)
        self._send(frame)

    def _ack(self, seq: int, ok: bool = True, error: str | None = None) -> None:
        frame: dict = {"t": "ack", "seq": int(seq), "ok": ok}
        if error is not None:
            frame["error"] = error
        self._send(frame)

    # ── inbound (control) ────────────────────────────────────────────────────
    def _read_loop(self) -> None:
        """Background reader: parse NDJSON control frames onto the queue.

        Never applies anything — application happens in ``drain_control`` on the
        trainer thread at the step boundary.
        """
        sock = self._sock
        if sock is None:
            return
        buf = b""
        while not self._closed:
            try:
                chunk = sock.recv(_RECV_CHUNK)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                if isinstance(frame, dict):
                    self._control_q.put(frame)

    def drain_control(self) -> list[dict]:
        """Return all queued inbound control frames (FIFO), clearing the queue.

        Called by the trainer each step. The trainer then applies them at the
        same boundary as the file-based live config.
        """
        out: list[dict] = []
        while True:
            try:
                out.append(self._control_q.get_nowait())
            except queue.Empty:
                break
        return out

    def note_applied_seq(self, seq: int) -> None:
        self.applied_seq = max(self.applied_seq, int(seq))

    def ack(self, seq: int, ok: bool = True, error: str | None = None) -> None:
        """Public per-command ack (status.appliedSeq is authoritative for set_hp)."""
        self._ack(seq, ok=ok, error=error)
