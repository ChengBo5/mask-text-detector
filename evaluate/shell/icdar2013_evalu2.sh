#!/bin/bash

py_dir="tools/test.py"
cfg_dir="evaluate/my_config/2_mask_rcnn/2_mask_rcnn_ICDAR2013.py"
weight_dir=("work_dirs/2_mask_rcnn_ICDAR2013/epoch_50.pth" 
            "work_dirs/2_mask_rcnn_ICDAR2013/epoch_70.pth" 
            "work_dirs/2_mask_rcnn_ICDAR2013/epoch_100.pth" 
            "work_dirs/2_mask_rcnn_ICDAR2013/epoch_150.pth" 
            "work_dirs/2_mask_rcnn_ICDAR2013/epoch_200.pth"
            "work_dirs/2_mask_rcnn_ICDAR2013/epoch_250.pth" 
            "work_dirs/2_mask_rcnn_ICDAR2013/epoch_300.pth")

log_dir="dataset/result/txt/2_mask_rcnn_ICDAR2013.txt"

fcos_score_thr_list=0.1
fcos_iou_threshold_list=(0.9)
rcnn_score_thr_list=(0.15)
rcnn_iou_threshold_list=(0.3)

for fcos_score_thr in ${fcos_score_thr_list[@]};
do
    for rcnn_iou_threshold in ${rcnn_iou_threshold_list[@]};
    do
        for rcnn_score_thr in ${rcnn_score_thr_list[@]};
        do
            for weight_dir_ in ${weight_dir[@]};
            do
                python $py_dir $cfg_dir $weight_dir_ --show-dir dataset/result/image_result

                echo "*************rcnn_score_thr="$weight_dir_ >&1 | tee -a $log_dir ;
                zip -jmq dataset/result/zip/icdar2013_rpn_box.zip dataset/result/icdar2013_rpn_box/* ;
                python evaluate/ICDAR2013/script.py -g=evaluate/ICDAR2013/gt.zip -s=dataset/result/zip/icdar2013_rpn_box.zip >&1 | tee -a $log_dir ;
                echo -e >&1 | tee -a $log_dir ;

                zip -jmq dataset/result/zip/icdar2013_mask_box.zip dataset/result/icdar2013_mask_box/* ;
                python evaluate/ICDAR2013/script.py -g=evaluate/ICDAR2013/gt.zip -s=dataset/result/zip/icdar2013_mask_box.zip >&1 | tee -a $log_dir ;
                echo -e >&1 | tee -a $log_dir ;

            done
            echo -e >> $log_dir
            echo -e >> $log_dir
            echo -e >> $log_dir
        done
    done
done
