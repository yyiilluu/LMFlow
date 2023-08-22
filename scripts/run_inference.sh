#!/bin/bash
limit=300

#output_models/ft_lora_7d_multi_reply_with_org_facebook/galactica-1.3b_20230504_202858
#output_models/ft_lora_7d_multi_reply_with_org_andreaskoepf/pythia-1.4b-gpt4all-pretrain_20230505_151335
#output_models/ft_lora_7d_multi_reply_v1_templates_andreaskoepf/pythia-1.4b-gpt4all-pretrain_20230511_150100
#output_models/ft_lora_7d_multi_reply_v1_templates_andreaskoepf/pythia-1.4b-gpt4all-pretrain_20230511_235022
#output_models/ft_lora_v2_3m_multi_reply_with_org_andreaskoepf/pythia-1.4b-gpt4all-pretrain_20230529_192026


init_model_name=andreaskoepf/pythia-1.4b-gpt4all-pretrain

# which data to predict on
#task_split=collections/1d_multi_reply_with_org/dev
task_split=test/agent_first_reply_with_org/text2text
data_version=v2
# Set the source directory
source_dir=../data_collection/data/$data_version/$task_split


# Loop through the list of files and copy each file to the destination directory
for file_path in $source_dir/*text2text.json;
do
    filename="$(basename $file_path)"
    echo "Predicting file $source_dir/$filename";
    CUDA_VISIBLE_DEVICES=0 \
        deepspeed examples/predict.py \
        --model_name_or_path $init_model_name \
#        --lora_model_path $model_path/${model_name} \
        --dataset_path ../data_collection/data/$data_version/$task_split/$filename \
        --max_eval_samples $limit \
        --backend json \
        --output_dir ../data_collection/data/$data_version/predictions/$task_split/${filename}_${limit}_${model_name} \
        --deepspeed examples/ds_config.json \

done