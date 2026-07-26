#!/bin/bash

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [ ! -d cache ]; then
  mkdir cache
fi

mkdir -p logs

echo "🚀 Starting Full Model Experiment minimal reproduction"

run_ablation() {
    local device=$1
    local runseed=$2
    local exp_name=$3
    local use_projector=$4
    local use_ot=$5
    local use_progressive_mask=$6
    local use_graph_expert=$7
    local proj_dropout=$8
    local epochs=$9
    local proj_blocks=${10:-2}
    local proj_act=${11:-relu}
    
    local few_shot_number=128 
    
    echo "▶️ [GPU $device] Starting: $exp_name (runseed=$runseed)"

    local out_file_prefix="cache/ablation/seed${runseed}/"
    mkdir -p "$out_file_prefix"
    local outfile="${out_file_prefix}${exp_name}.csv"

    #local out_file_prefix="cache/"
    
    local dataset_list=(bace hiv bbbp tox21 esol freesolv)
    
    for dataset in "${dataset_list[@]}"; do
        CUDA_VISIBLE_DEVICES=$device python downstream_test.py \
            --use_projector $use_projector \
            --use_ot_alignment $use_ot \
            --use_progressive_mask $use_progressive_mask \
            --use_graph_expert $use_graph_expert \
            --projector_dropout $proj_dropout \
            --projector_blocks $proj_blocks \
            --projector_act $proj_act \
            --few_shot $few_shot_number \
            --few_shot_fashion innovation \
            --transformer_backbone gimlet \
            --model_name_or_path "haitengzhao/gimlet" \
            --tokenizer_name t5-small \
            --dataset "$dataset" \
            --runseed "$runseed" \
            --batch_size 128 \
            --grad_accum_step 8 \
            --lr 1e-2 \
            --epochs $epochs \
            --test_interval 20 \
            --transform_in_collator \
            --no_eval_train \
            --output_result_to_file $outfile \
            --not_retest_tasks_in_result_file
    done
    echo "✅ [GPU $device] Finished: $exp_name"
}

DROP=0.4
EP=60
PREFIX="Drop${DROP}_Ep${EP}"

run_seed_batch() {
    local runseed=$1

    echo "🚀 Starting runseed=$runseed on GPU 7"

    # 所有消融实验串行执行
    run_ablation 7 "$runseed" "${PREFIX}_wo_Graph_Expert" \
        True True True False "$DROP" "$EP" \
        > "logs/ablation_gpu7_seed${runseed}_${PREFIX}_wo_Graph_Expert.log" 2>&1

    run_ablation 7 "$runseed" "${PREFIX}_wo_Dynamic_Masking" \
        True True False True "$DROP" "$EP" \
        > "logs/ablation_gpu7_seed${runseed}_${PREFIX}_wo_Dynamic_Masking.log" 2>&1

    run_ablation 7 "$runseed" "${PREFIX}_wo_OT_Alignment" \
        True False True True "$DROP" "$EP" \
        > "logs/ablation_gpu7_seed${runseed}_${PREFIX}_wo_OT_Alignment.log" 2>&1

    echo "✅ Finished runseed=$runseed"
}

# 种子 5、6、7、8、9 串行执行
for runseed in 5 6 7 8 9; do
    run_seed_batch "$runseed"
done