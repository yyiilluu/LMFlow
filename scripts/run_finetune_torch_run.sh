#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=pretraining_pythia_1b
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

dataset_path=/home/ec2-user/SageMaker/repos/test_data/processed
# dataset_path=/home/ec2-user/SageMaker/repos/small_data

mkdir -p ${output_dir} ${log_dir}

torchrun --nproc_per_node 8 examples/finetune.py \
    --model_name_or_path EleutherAI/pythia-6.9b-deduped \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --block_size 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --fp16 \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --report_to="none" \
    --do_train \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap GPTNeoXLayer \
    --gradient_checkpointing True \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err