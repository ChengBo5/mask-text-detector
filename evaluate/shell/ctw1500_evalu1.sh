#!/bin/bash

py_dir="tools/test.py"
cfg_dir="evaluate/config/9mask_gfl/9mask_gfl_ctw2015.py"
weight_dir=("work_dirs/9mask_gfl_ctw2015/epoch_100.pth"
            "work_dirs/9mask_gfl_ctw2015/epoch_130.pth"
            "work_dirs/9mask_gfl_ctw2015/epoch_170.pth"
            "work_dirs/9mask_gfl_ctw2015/epoch_200.pth"
            "work_dirs/9mask_gfl_ctw2015/epoch_250.pth"
            "work_dirs/9mask_gfl_ctw2015/epoch_300.pth")

log_dir="evaluate/config/9mask_gfl/ctw2015.txt"

fcos_score_thr_list=0.1
fcos_iou_threshold_list=(0.9)
rcnn_score_thr_list=(0.3)
rcnn_iou_threshold_list=(0.3)

for fcos_score_thr in ${fcos_score_thr_list[@]};
do
    for fcos_iou_threshold in ${fcos_iou_threshold_list[@]};
    do
        for rcnn_score_thr in ${rcnn_score_thr_list[@]};
        do
            for weight_dir_ in ${weight_dir[@]};
            do 
                python $py_dir $cfg_dir $weight_dir_ --eval bbox --show-dir evaluate/image_result

                echo "*************rcnn_score_thr="$weight_dir_ >&1 | tee -a $log_dir ;
                python evaluate/ctw1500_evalu/eval_ctw1500.py >&1 | tee -a $log_dir
                echo -e >&1 | tee -a $log_dir
                rm -rf evaluate/ctw1500_evalu/predict1/*
            done
            echo -e >> $log_dir
            echo -e >> $log_dir
            echo -e >> $log_dir
        done
    done
done
