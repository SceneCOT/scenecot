#!/bin/bash

# ------------------------------
# Reproducibility env settings
# ------------------------------
export SCENECOT_EXP_ROOT=${SCENECOT_EXP_ROOT:-./outputs}
export SCENECOT_DATA_ROOT=${SCENECOT_DATA_ROOT:-./data_assets}
export SCENECOT_COT_DATA_ROOT=${SCENECOT_COT_DATA_ROOT:-${SCENECOT_DATA_ROOT}/scenecot_cot_data}
export HF_HOME=${HF_HOME:-./.cache/huggingface}

# model sources
export SCENECOT_LLM_PATH=${SCENECOT_LLM_PATH:-liuhaotian/llava-v1.5-7b}
export SCENECOT_VISION_TOWER_PATH=${SCENECOT_VISION_TOWER_PATH:-openai/clip-vit-large-patch14-336}
export SCENECOT_PQ3D_TOKENIZER_PATH=${SCENECOT_PQ3D_TOKENIZER_PATH:-openai/clip-vit-large-patch14}

# optional PQ3D checkpoints
export SCENECOT_POINTNET_TOKENIZER_PATH=${SCENECOT_POINTNET_TOKENIZER_PATH:-}
export SCENECOT_QUERY3D_PRETRAIN_PATH=${SCENECOT_QUERY3D_PRETRAIN_PATH:-}

# optional annotation paths (HF SceneCOT COT data layout)
# - ${SCENECOT_COT_DATA_ROOT}/MSQA contains situated_qa_{train,val,test}_pure_txt.json
# - ${SCENECOT_COT_DATA_ROOT}/GQA3D contains gqa3d_{train,val,test}.json
export SCENECOT_MSR3D_ANNO_DIR=${SCENECOT_MSR3D_ANNO_DIR:-${SCENECOT_COT_DATA_ROOT}/MSQA}
export SCENECOT_GQA3D_ANNO_DIR=${SCENECOT_GQA3D_ANNO_DIR:-${SCENECOT_COT_DATA_ROOT}/GQA3D}

# optional logging mode (set to online if needed)
export WANDB_MODE=${WANDB_MODE:-disabled}

CUDA_LAUNCH_BLOCKING=1 python launch.py --name scene_cot \
                 --qos lv0b \
                 --mem_per_gpu 100 \
                 --cpu_per_task 16 \
                 --time 48 \
                 --config configs/default_cot.yaml \
                 --port 2021 \
                 --gpu_per_node 4 \
                 --num_nodes 1 \
                 --partition HGX \
                 --exclude dgx-hyperplane12 \
                model=SceneCOTAgent \
                task=scenecot_scanent_msqa_gqa3d \
                note=main_msqa_gqa3d \
                grounding.enable=True \
                grounding.loss_type=ce \
                grounding.grd_loss_weight=0.1 \
                grounding.grd_text_hidden_states=average_embedding \
                grounding.use_region_mask=True \
                cot.mask_obj_prob_loc_token=True \
                cot.use_oracle_obj_content=False \
                cot.cot_no_scene_tokens=True \
                cot.gqa3d_val_set_size=500 \
                cot.msr3d_val_set_size=1000 \
                llm=llava1.5-7b \
                llm.max_out_len=512 \
                llm.max_context_len=1024 \
                vision3d.name=PQ3D \
                vision3d.use_embodied_token=False \
                debug.flag=False \
                task.qacot.QACOTScanNetMSR3D.evaluator=MSQACOTEvaluator \
                task.leomix.LeoMix.debug_size=40000000 \
                task.training.epochs=5 \
                data.cotqa.msr3d.pc_type=pred \
                data.cotqa.gqa3d.pc_type=gt \
                data.nms_iou_threshold=1 \
                data.cotqa.use_pred_for_train=False \
                data.cotqa.msr3d.anno_dir=${SCENECOT_MSR3D_ANNO_DIR} \
                data.cotqa.gqa3d.anno_dir=${SCENECOT_GQA3D_ANNO_DIR} \
                dataloader.train.batchsize=2 \
                dataloader.eval.batchsize=2 \
                dataloader.eval.num_workers=1
