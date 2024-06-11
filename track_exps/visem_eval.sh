#!/usr/bin/env bash
#!sh track_exps/mot_train_val_mota.sh

GROUNDTRUTH=visem/train

OUTPUT_DIR=output_visem/exp_0415_sche4to1_ep100

RESULTS=${OUTPUT_DIR}/val/tracks
GT_TYPE=_val_half
THRESHOLD=-1

#validation phase
python3 main_track.py  \
--output_dir ${OUTPUT_DIR} \
--dataset_file visem \
--coco_path visem \
--batch_size 1 \
--resume ${OUTPUT_DIR}/checkpoint.pth \
--eval \
--track_eval_split val \
--with_box_refine \
--num_queries 500 \
--det_val
