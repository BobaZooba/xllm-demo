#!/bin/bash

deepspeed --num_gpus=8 xllm_demo/cli/train.py \
  --use_gradient_checkpointing True \
  --deepspeed_stage 2 \
  --stabilize True \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --use_flash_attention_2 False \
  --load_in_4bit True \
  --apply_lora True \
  --raw_lora_target_modules all \
  --per_device_train_batch_size 8 \
  --warmup_steps 1000 \
  --save_total_limit 0 \
  --push_to_hub True \
  --hub_model_id BobaZooba/DemoXLLM7B-v1-LoRA \
  --hub_private_repo True \
  --report_to_wandb True \
  --logging_steps 1 --num_train_epochs 3 \
  --save_steps 1000 \
  --max_steps 10050 \
  --path_to_env_file ./.env \
  --collator_key last_part \
  --trainer_key steps \
  --experiment_key check_model
