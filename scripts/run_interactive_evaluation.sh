#!/bin/bash

# --model_name_or_path specifies the original huggingface model
# --lora_model_path specifies the model difference introduced by finetuning,
#   i.e. the one saved by ./scripts/run_finetune_with_lora.sh

deepspeed examples/interactive_evaluation.py \
    --answer_type text \
    --cache_dir /home/ec2-user/SageMaker/repos/LMFlow/cache \
    --model_name_or_path /home/ec2-user/SageMaker/repos/LMFlow/output_models/pretraining_pythia_1b_ift_compress_decompress_16k \
    --deepspeed examples/ds_config.json \
#    --lora_model_path /home/ec2-user/SageMaker/repos/LMFlow/output_models/finetune_alpaca_with_lora \