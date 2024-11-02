python main_train.py \
--image_dir /data2/liuxiaocong/oph/RepGen-main/data/zju2/images/ \
--ann_path /data2/liuxiaocong/oph/RepGen-main/data/zju2/annotation.json \
--dataset_name zju2 \
--max_seq_length 40 \
--use_jieba True \
--n_gpu 1 \
--threshold 3 \
--batch_size 16 \
--epochs 20 \
--save_dir results/zju2-36 \
--step_size 10 \
--gamma 0.1 \
--contra_type 'base' \
--finetune_lambda 1 \
--contra_lambda 0.2 \
--contra_temperature 0.1 \
--contra_embed_size 256 \
--contra_epoch 0 \
--seed 36

#--resume model/mimic_cxr.pth \