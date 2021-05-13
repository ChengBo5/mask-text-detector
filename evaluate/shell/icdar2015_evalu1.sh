#!/bin/bash

py_dir="tools/test.py"
cfg_dir="evaluate/config/9mask_gfl/9mask_gfl.py"
weight_dir="work_dirs/9mask_gfl/epoch_160.pth"
log_dir="evaluate/config/9mask_gfl/icdar2015_后期减少rpn权重.txt"

fcos_score_thr_list=0.1
fcos_iou_threshold_list=(0.9)
rcnn_score_thr_list=(0.05 0.1 0.15 0.2)
rcnn_iou_threshold_list=(0.3)

for fcos_score_thr in ${fcos_score_thr_list[@]};
do
    for fcos_iou_threshold in ${fcos_iou_threshold_list[@]};
    do
        for rcnn_score_thr in ${rcnn_score_thr_list[@]};
        do
            for rcnn_iou_threshold in ${rcnn_iou_threshold_list[@]};
            do 
                # python $py_dir $cfg_dir $weight_dir --eval bbox --show-dir evaluate/image_result --cfg-options \
                #         test_cfg.rpn.nms.iou_threshold=$fcos_iou_threshold \
                #         test_cfg.rcnn.score_thr=$rcnn_score_thr \
                #         test_cfg.rcnn.nms.iou_threshold=$rcnn_iou_threshold

                python $py_dir $cfg_dir $weight_dir --eval bbox --show-dir evaluate/image_result --cfg-options \
                        test_cfg.rcnn.score_thr=$rcnn_score_thr \
                        test_cfg.rcnn.nms.iou_threshold=$rcnn_iou_threshold

                echo "*************rcnn_score_thr="$weight_dir_ >&1 | tee -a $log_dir ;
                zip -jmq evaluate/icdar2015_evalu/rpn_box.zip evaluate/icdar2015_evalu/rpn_box/* ;
                python evaluate/icdar2015_evalu/script.py -g=evaluate/icdar2015_evalu/gt.zip -s=evaluate/icdar2015_evalu/rpn_box.zip >&1 | tee -a $log_dir ;
                echo -e >&1 | tee -a $log_dir ;

                zip -jmq evaluate/icdar2015_evalu/box.zip evaluate/icdar2015_evalu/box/* ;
                python evaluate/icdar2015_evalu/script.py -g=evaluate/icdar2015_evalu/gt.zip -s=evaluate/icdar2015_evalu/box.zip >&1 | tee -a $log_dir ;
                echo -e >&1 | tee -a $log_dir ;

                zip -jmq evaluate/icdar2015_evalu/mask_box.zip evaluate/icdar2015_evalu/mask_box/* ;
                python evaluate/icdar2015_evalu/script.py -g=evaluate/icdar2015_evalu/gt.zip -s=evaluate/icdar2015_evalu/mask_box.zip >&1 | tee -a $log_dir ;
                echo -e >&1 | tee -a $log_dir ;
            done
            echo -e >> $log_dir
            echo -e >> $log_dir
            echo -e >> $log_dir
        done
    done
done
