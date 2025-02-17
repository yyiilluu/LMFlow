#!/bin/bash
# Please run this script under ${project_id} in project directory of

# Parses arguments
model_name_or_path=gpt2
dataset_path=data/alpaca/train
output_dir=output_models/finetune
deepspeed_args="--master_port=11000"

exp_id=finetune_wa_t2t_with_lora
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}

dataset_path=${project_dir}/data/wa/training_files/text2text

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path facebook/galactica-1.3b \
    --cache_dir /home/ec2-user/SageMaker/repos/LMFlow/cache \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --block_size 512 \
    --per_device_train_batch_size 4 \
    --use_lora 1 \
    --lora_r 8 \
    --save_aggregated_lora 0\
    --deepspeed configs/ds_config_zero2.json \
    --fp16 \
    --run_name finetune_with_lora \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --report_to="none" \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
