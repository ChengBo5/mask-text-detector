#!/bin/bash

py_dir="tools/test.py"
cfg_dir="evaluate/my_config/3anchor-free/3_ICDAR2015_anchor_free.py"
weight_dir=("work_dirs/3_ICDAR2015_anchor_free/epoch_50.pth" 
            "work_dirs/3_ICDAR2015_anchor_free/epoch_70.pth" 
            "work_dirs/3_ICDAR2015_anchor_free/epoch_120.pth" 
            "work_dirs/3_ICDAR2015_anchor_free/epoch_150.pth" 
            "work_dirs/3_ICDAR2015_anchor_free/epoch_200.pth"
            "work_dirs/3_ICDAR2015_anchor_free/epoch_250.pth" 
            "work_dirs/3_ICDAR2015_anchor_free/epoch_300.pth")

log_dir="dataset/result/txt/anchor_free_icdar2015.txt"

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
                python $py_dir $cfg_dir $weight_dir_ --eval bbox --show-dir dataset/result/image_result\
                        # --cfg-options \
                        # test_cfg.rcnn.score_thr=$rcnn_score_thr \
                        # test_cfg.rcnn.nms.iou_threshold=$rcnn_iou_threshold

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
