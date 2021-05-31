#!/bin/bash

py_dir="tools/test.py"
cfg_dir="evaluate/my_config/5_joint_regression/5_joint_regression_ICDAR2015.py"
weight_dir="work_dirs/5_joint_regression_ICDAR2015/epoch_300.pth"
log_dir="dataset/result/txt/5_joint_regression_ICDAR2015.txt"

fcos_score_thr_list=0.1
fcos_iou_threshold_list=(0.9)
rcnn_score_thr_list=(0.65 0.7 0.75)
rcnn_iou_threshold_list=(0.3)

for fcos_score_thr in ${fcos_score_thr_list[@]};
do


    for fcos_iou_threshold in ${fcos_iou_threshold_list[@]};
    do
        for rcnn_score_thr in ${rcnn_score_thr_list[@]};
        do
            for rcnn_iou_threshold in ${rcnn_iou_threshold_list[@]};
            do 
                python $py_dir $cfg_dir $weight_dir --show-dir dataset/result/image_result --cfg-options \
                        model.test_cfg.rpn.nms.iou_threshold=$fcos_iou_threshold \
                        model.test_cfg.rcnn.score_thr=$rcnn_score_thr \
                        model.test_cfg.rcnn.nms.iou_threshold=$rcnn_iou_threshold


                echo "*************rcnn_score_thr="$rcnn_score_thr >&1 | tee -a $log_dir ;
                # zip -jmq evaluate/icdar2015_evalu/rpn_box.zip evaluate/icdar2015_evalu/rpn_box/* ;
                # python evaluate/icdar2015_evalu/script.py -g=evaluate/icdar2015_evalu/gt.zip -s=evaluate/icdar2015_evalu/rpn_box.zip >&1 | tee -a $log_dir ;
                # echo -e >&1 | tee -a $log_dir ;

                zip -jmq dataset/result/zip/icdar2015_box.zip dataset/result/icdar2015_box/* ;
                python evaluate/ICDAR2015/script.py -g=evaluate/ICDAR2015/gt.zip -s=dataset/result/zip/icdar2015_box.zip >&1 | tee -a $log_dir ;
                echo -e >&1 | tee -a $log_dir ;

                zip -jmq dataset/result/zip/icdar2015_mask_box.zip dataset/result/icdar2015_mask_box/* ;
                python evaluate/ICDAR2015/script.py -g=evaluate/ICDAR2015/gt.zip -s=dataset/result/zip/icdar2015_mask_box.zip >&1 | tee -a $log_dir ;
                echo -e >&1 | tee -a $log_dir ;
            done
            echo -e >> $log_dir
            echo -e >> $log_dir
            echo -e >> $log_dir
        done
    done
done
