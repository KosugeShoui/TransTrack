#!/usr/bin/env bash
#!sh track_exps/mot_train_val_mota.sh
DATAFILE=visem

GROUNDTRUTH1=${DATAFILE}/train
GROUNDTRUTH2=${DATAFILE}/test

OUTPUT_DIR=output_visem/exp_0416_sche4to2_ep100

RESULTS1=${OUTPUT_DIR}/val/tracks
RESULTS2=${OUTPUT_DIR}/test/tracks
GT_TYPE1=_val_half
GT_TYPE2=test
THRESHOLD=-1

"""
#validation phase train val data
python3 main_track.py  \
--output_dir ${OUTPUT_DIR} \
--dataset_file ${DATAFILE} \
--coco_path ${DATAFILE} \
--batch_size 1 \
--resume ${OUTPUT_DIR}/checkpoint.pth \
--eval \
--track_eval_split val \
--with_box_refine \
--num_queries 500

#validation phase test data
python3 main_track.py  \
--output_dir ${OUTPUT_DIR} \
--dataset_file ${DATAFILE} \
--coco_path ${DATAFILE} \
--batch_size 1 \
--resume ${OUTPUT_DIR}/checkpoint.pth \
--eval \
--with_box_refine \
--num_queries 500
"""

#eval phase(validation)
python3 track_tools/eval_motchallenge.py \
--groundtruths ${GROUNDTRUTH1} \
--tests ${RESULTS1} \
--gt_type ${GT_TYPE1} \
--eval_official \
--score_threshold ${THRESHOLD}

#eval phase(test)
python3 track_tools/eval_motchallenge.py \
--groundtruths ${GROUNDTRUTH2} \
--tests ${RESULTS2} \
--gt_type ${GT_TYPE2} \
--eval_official \
--score_threshold ${THRESHOLD}

python3 util/print_exp.py ${OUTPUT_DIR}



