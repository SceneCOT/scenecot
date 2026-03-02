#!/bin/bash
CUDA_LAUNCH_BLOCKING=1 python launch.py --name scene_cot \
                 --qos lv0b \
                 --mem_per_gpu 100 \
                 --cpu_per_task 16 \
                 --time 48 \
                 --config configs/default_cot.yaml \
                 --port 2021 \
                 --gpu_per_node 4 \
                 --num_nodes 1 \
                 --partition DGX \
                 --mode python \
                model=SceneCOTAgent \
                task=scenecot_scanent_msqa_gqa3d \
                note=msqa_beacon3d_test_moe \
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
                data.cotqa.msr3d.anno_dir="" \
                data.cotqa.gqa3d.anno_dir="" \
                dataloader.train.batchsize=2 \
                dataloader.eval.batchsize=2 \
                dataloader.eval.num_workers=1 \
                ++moe.enable=True \
                ++moe.expert1_path="" \
                ++moe.expert2_path="" \
                ++moe.expert1_trigger_types="[counting,existence]" \
                pretrained_ckpt_path="" \
                mode=test
