# Plan: ICE Training Pipeline — Ready to Run

## Context

ICE label generation is running in parallel (~4 days). While waiting, we implement the ICE training loop so it's ready to run immediately once labels finish. This is Track 2, item 2 from `04_next_steps.md`.

The ICE model architecture (`ICE` 1D CNN), dataset (`ICEDataset`), config (`configs/training/ice.yaml`), LR schedule (`cosine_with_warmup`), and gradient utils are all implemented. What's missing: `BaseTrainer.train()`, `ICETrainer.train_step()/evaluate()`, checkpointing functions, train.py dispatch, and a collation function for variable-length sequences.

## Changes

### 1. Implement checkpointing (`src/bgkit/training/checkpointing.py`)

Simple torch-based save/load (no Accelerate needed for ICE — single GPU, tiny model):

- `save_checkpoint()`: Create timestamped dir, `torch.save()` each state_dict, write `metadata.json`
- `load_checkpoint()`: Read metadata, load state dicts

### 2. Implement `BaseTrainer.train()` loop (`src/bgkit/training/base_trainer.py`)

Keep it minimal — no Accelerate for now (ICE trains on one GPU with bf16 autocast). Add Accelerate later for Phase 1/2 when we need gradient accumulation across large models.

**BaseTrainer gets:**
- `train()`: Main loop — iterate DataLoader, call `train_step()`, log to wandb, eval every N steps, checkpoint every M steps, LR scheduling
- `save_checkpoint()` / `load_checkpoint()`: Delegate to checkpointing module
- Abstract: `train_step()`, `evaluate()`, `setup()` (subclass creates model/optimizer/dataset)

**Training loop pseudocode:**
```
setup()  # subclass creates model, optimizer, dataloader
for step in range(max_steps):
    lr = cosine_with_warmup(step, ...)
    set_lr(optimizer, lr)
    batch = next(dataloader_iter)
    metrics = train_step(batch)
    log_to_wandb(metrics, step)
    if step % eval_every == 0: evaluate()
    if step % save_every == 0: save_checkpoint()
```

### 3. Implement `ICETrainer` (`src/bgkit/training/ice_trainer.py`)

**`setup()`:**
- Load frozen Qwen3-Embedding-0.6B (`AutoModel.from_pretrained`, eval mode, no grad)
- Create ICE model from config
- Create ICEDataset from `cfg.data.ice_labels.output_dir`
- Create DataLoader with padding collator
- Create AdamW optimizer for ICE params only

**`train_step(batch)`:**
1. Pad batch (token_ids, ce_values) — variable-length sequences
2. Forward frozen embedding model: `token_ids → embeddings (B, L, 1024)`
3. Forward ICE: `embeddings → ce_pred (B, L)`
4. Align shapes: `ce_pred[:, :-1]` vs `ce_values` (CE has N-1 values for N tokens)
5. MSE loss (masked for padding)
6. Uniformity regularizer: `-variance(ce_pred)` weighted by `uniformity_reg_weight=0.1`
7. Backward, clip grads, step optimizer
8. Return metrics dict

**`evaluate()`:**
- Run over eval split (10% holdout or separate shard)
- Compute MSE, Pearson correlation, prediction variance
- Return metrics dict

**Collation function** (in `ice_trainer.py` or `src/bgkit/data/collators.py`):
- Pad `token_ids` to max length in batch with pad_id=0
- Pad `ce_values` to max length with 0.0
- Create `attention_mask` (1 for real tokens, 0 for pad)
- Return dict of tensors

### 4. Wire `train.py` dispatch (`scripts/train.py`)

Add ICE phase:
```python
if phase == "ice":
    from bgkit.training.ice_trainer import ICETrainer
    trainer = ICETrainer(cfg)
    trainer.train()
```

### 5. Add unit tests (`tests/unit/training/test_ice_trainer.py`)

- Test ICE collation function: variable-length inputs → padded tensors + mask
- Test `train_step()` with tiny mock embedding model + small ICE: runs without error, returns expected metric keys
- Test that loss decreases over a few steps on synthetic data
- Test checkpoint save/load roundtrip

## Files Modified

| File | Action |
|---|---|
| `src/bgkit/training/checkpointing.py` | Implement save/load |
| `src/bgkit/training/base_trainer.py` | Implement `train()` loop, `save/load_checkpoint()` |
| `src/bgkit/training/ice_trainer.py` | Implement `setup()`, `train_step()`, `evaluate()`, collator |
| `scripts/train.py` | Add ICE phase dispatch |
| `tests/unit/training/test_ice_trainer.py` | New: collation, train_step, loss, checkpoint tests |

## Verification

1. `make test` — all existing tests still pass
2. New tests pass: `.venv/bin/pytest tests/unit/training/ -v`
3. Dry run with 10 synthetic shards (mock data, no real labels needed):
   - Instantiate ICETrainer with small config
   - Verify train_step runs and returns metrics
4. Once ICE labels finish: run in Docker container:
   ```bash
   docker compose -f docker/docker-compose.yaml run train \
     scripts/train.py training=ice data.ice_labels.output_dir=./data/processed/ice_labels
   ```
