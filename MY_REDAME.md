# 1.配置环境

可以按照mmdection官方文档配。我们选用的
```
mmcv==1.3.3
mmdet==2.12.0
torch==1.7.0
torchvision==0.8.0
cudatoolkit==10.2.89
```

我们提供以下简易安装命令
```
conda create -n mask-text-detector python=3.7 -y
conda activate mask-text-detector
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install mmcv-full==latest+torch1.7.0+cu102 -f https://download.openmmlab.com/mmcv/dist/index.html
git clone https://github.com/chengbo/mask-text-detector.git
cd mask-text-detector
pip install -r requirements/build.txt
pip install -v -e .
```

# 2训练数据下载








# 3训练和测试模型


- ICDAR2015

  首先在上一步下载完数据集,运行训练代码。

    ```
    conda activate mask-text-detector
    #单gpu训练
    python tools/train.py evaluate/my_config/2_mask_rcnn/2_mask_rcnn_ICDAR2013.py
    #多gpu训练 
    # bash tools/dist_train.sh evaluate/my_config/2_mask_rcnn/2_mask_rcnn_ICDAR2013.py [gpu_num]
    bash tools/dist_train.sh evaluate/my_config/2_mask_rcnn/2_mask_rcnn_ICDAR2013.py 4
    ```

- ICDAR2013   
    ```
    conda activate mask-text-detector
    #单gpu训练
    python tools/train.py evaluate/my_config/2_mask_rcnn/2_mask_rcnn_ICDAR2013.py
    #多gpu训练 
    # bash tools/dist_train.sh evaluate/my_config/2_mask_rcnn/2_mask_rcnn_ICDAR2013.py [gpu_num]
    bash tools/dist_train.sh evaluate/my_config/2_mask_rcnn/2_mask_rcnn_ICDAR2013.py 4
    ```






