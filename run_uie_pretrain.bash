#!/usr/bin/env bash
# -*- coding:utf-8 -*-

EXP_ID=$(date +%F-%H-%M-$RANDOM)
export CUDA_VISIBLE_DEVICES="0"
export batch_size="64"
export model_name=t5-v1_1-base
export data_name=pretrain_data
export lr=1e-4
export task_name="record"
export begin_index=1
export run_time="1"
export seed="42"
export lr_scheduler=linear
export label_smoothing="0"
export epoch=30
export decoding_format='spotasoc'
export eval_steps=1000000
export warmup_ratio=0.06
export constraint_decoding=''
export verbose=false
export preprocess=False
source scripts/function_code.bash

model_folder=output/Pretrain_${EXP_ID}_${model_name_log}_${decoding_format}_${data_name}_${lr_scheduler}_lr${lr}_ls${label_smoothing}_${batch_size}_wu${warmup_steps}
data_folder=pretrain_data/${data_name}

output_dir=${model_folder}

if [[ ${verbose} == true ]]
then
    stdout_file=/dev/stdout
    stderr_file=/dev/stderr
    disable_tqdm=False
else
    stdout_file=${output_dir}.log
    stderr_file=${output_dir}.log
    disable_tqdm=True
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${run_command} \
    run_uie_pretrain.py \
    --do_train --do_eval ${fp16} \
    --preprocess=${preprocess} \
    --preprocessed_folder=pretrain_data/${data_name}_preprocessed \
    --use_fast_tokenizer=True \
    --evaluation_strategy=${evaluation_strategy} \
    --metric_for_best_model eval_loss \
    --greater_is_better=False \
    --save_total_limit 20 \
    --load_best_model_at_end \
    --max_source_length=128 \
    --max_target_length=128 \
    --num_train_epochs=${epoch} \
    --task=${task_name} \
    --train_file=${data_folder}/train.json \
    --validation_file=${data_folder}/val.json \
    --record_schema=${data_folder}/${task_name}.schema \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=$((batch_size * 4)) \
    --output_dir=${output_dir} \
    --logging_dir=${output_dir}_log \
    --model_name_or_path=${model_name} \
    --learning_rate=${lr} \
    --lr_scheduler_type=${lr_scheduler} \
    --label_smoothing_factor=${label_smoothing} \
    --eval_steps ${eval_steps} \
    --decoding_format ${decoding_format} \
    --warmup_ratio ${warmup_ratio} \
    --source_prefix="" \
    --preprocessing_num_workers=${preprocessing_num_workers:-"16"} \
    --dataloader_num_workers=4 \
    --ddp_find_unused_parameters=False \
    --seed=${seed}${index} \
    --no_remove_unused_columns \
    --predict_with_generate=False \
    --skip_memory_metrics \
    --sortish_sampler \
    --meta_negative=${negative} \
    --meta_positive_rate=${positive} \
    --ordered_prompt=${ordered_prompt} \
    --disable_tqdm=${disable_tqdm} >${stdout_file} 2>${stderr_file}
