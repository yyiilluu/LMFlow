#!/bin/bash

# --model_name_or_path specifies the original huggingface model
# --lora_model_path specifies the model difference introduced by finetuning,
#   i.e. the one saved by ./scripts/run_finetune_with_lora.sh
CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/interactive_evaluation.py \
    --answer_type text \
    --model_name_or_path home/ec2-user/SageMaker/repos/LMFlow/output_models/finetune_alpaca_with_lora \
    --lora_model_path output_models/finetune_alpaca_with_lora_eval \
    --deepspeed examples/ds_config.json \
