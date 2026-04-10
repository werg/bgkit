"""Phase 2 knowledge-retrieval and injection training.

Training pipeline:
- Steps 1-4: KRTrainer (single KR trainer, step-driven via config)
- Step 5: KRStep5Trainer (extends KRTrainer with target LLM QLoRA injection)
- Tracks B/C: KRTrainer with git_kr / user_memory configs
"""
