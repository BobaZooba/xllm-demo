#!/bin/bash

python3 xllm_demo/cli/prepare.py \
  --text_field rejected \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --path_to_env_file ./.env \
  --collator_key last_part \
  --trainer_key steps \
  --experiment_key check_model
