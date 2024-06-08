python train_eval.py -d cifar100 \
    -m vit_base_patch16_224.augreg_in21k --head_dim_type task_classes --logit_type head_out --training_string prompt head \
    --prompt_len 4 -b 256 --temperature 28 \
    --null_thres_mode adaptive --null_eta1 0.97 --null_eta2 0.97 --use_null_space --data_root "DATA_ROOT" --seed 2024 $@
