#!/bin/bash

python3 xllm_demo/cli/quantize.py \
  --model_name_or_path ./fused_model/ \
  --apply_lora False --stabilize False \
  --quantized_model_path ./quantized_model/ \
  --prepare_model_for_kbit_training False \
  --quantized_hub_model_id DemoXLLM7B-v1-GPTQ \
  --quantized_hub_private_repo True \
  --low_cpu_mem_usage \
  --path_to_env_file ./.env \
  --collator_key last_part \
  --trainer_key steps \
  --experiment_key check_model
  