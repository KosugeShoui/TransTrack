#!/usr/bin/env bash
#!sh track_exps/mot_train_val_mota.sh
DATAFILE=visem

GROUNDTRUTH1=${DATAFILE}/train
GROUNDTRUTH2=${DATAFILE}/test

OUTPUT_DIR=output_visem/exp_0610_ep50_sche

RESULTS1=${OUTPUT_DIR}/val/tracks
RESULTS2=${OUTPUT_DIR}/test/tracks
GT_TYPE1=_val_half
GT_TYPE2=test
THRESHOLD=-1


#training phase
python3 -m torch.distributed.launch \
--nproc_per_node=1 \
--use_env main_track.py  \
--output_dir ${OUTPUT_DIR} \
--dataset_file ${DATAFILE} \
--coco_path ${DATAFILE} \
--batch_size 5  \
--with_box_refine  \
--num_queries 500 \
--set_cost_class 2 \
--set_cost_bbox 5 \
--set_cost_giou 4 \
--epochs 30 \
--lr_drop 100 \
--final_weight 1.0 \
--device cuda \
--loss_schedule 

# --det_val true
#--frozen_weights mot/619mot17_mot17.pth


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


#print exp name
python3 util/print_exp.py ${OUTPUT_DIR}

#visualize curve phase
python3 learning_curve.py ${OUTPUT_DIR}
python3 learning_curve_each.py ${OUTPUT_DIR}
python3 learning_unscaled_curve_each.py ${OUTPUT_DIR}
