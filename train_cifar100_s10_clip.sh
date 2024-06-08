python train_eval.py -d cifar100 \
    -m vit_base_patch16_clip_quickgelu_224.openai --head_dim_type text_dim --logit_type sim_imgtext --transform_type clip \
    --prompt_len 4 --prompt_end_block 3 --temperature 1 -et 2 -b 220 --lr 0.001 \
    --logit_scale_trainable True --training_string prompt logit_scale --refine_head True \
    --null_thres_mode adaptive --null_eta1 0.98 --null_eta2 0.98 --use_null_space --data_root "DATA_ROOT" --seed 2024 $@
