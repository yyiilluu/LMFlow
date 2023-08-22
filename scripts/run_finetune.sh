#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=pretraining_pythia_1b_ft_compression
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}


# model=EleutherAI/pythia-1b-deduped
# model=/home/ec2-user/SageMaker/repos/LMFlow/output_models/pretraining_pythia_1b/
model=EleutherAI/pythia-1b-deduped


## pretraining data
# dataset_path=/home/ec2-user/SageMaker/repos/test_data/processed
# dataset_path=/home/ec2-user/SageMaker/repos/small_data

## instruction tuning data
# dataset_path=/home/ec2-user/SageMaker/repos/v3_6m_ticket_reply

## openai paraphrasing tasks
#dataset_path=/home/ec2-user/SageMaker/repos/openai_data


## compression data
dataset_path=/home/yilu/workspace/repos/compressor/compressor/data/instruction_ft_data/ift_data

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model} \
    --cache_dir /home/ec2-user/SageMaker/repos/LMFlow/cache \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 4 \
    --learning_rate 2e-5 \
    --block_size 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --run_name finetune \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --report_to="none" \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
