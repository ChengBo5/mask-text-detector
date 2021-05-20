import numpy as np
import os
import cv2
import torch

def show_icdar2015_rpn_image(proposal_list, img):
    bbox_results = proposal_list[0].detach().cpu().numpy()
    polys = np.array(bbox_results).reshape(-1, 5)
    image_source = img.cpu().detach().numpy()
    image_source = np.transpose(image_source, (0, 2, 3, 1))
    mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1)
    std = np.array([58.395, 57.12, 57.375]).reshape(1, -1)
    image_source = np.multiply(image_source, std)
    image_source = np.add(image_source, mean).astype(np.uint8)
    image_source = image_source[0].copy()

    for id in range(polys.shape[0]):
        x1 = polys[id][0]
        y1 = polys[id][1]
        x2 = polys[id][2]
        y2 = polys[id][3]
        cv2.rectangle(image_source, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.namedWindow("AlanWang")
    cv2.imshow('AlanWang', image_source)
    cv2.waitKey(10)  # 显示 10000 ms 即 10s 后消失
    cv2.destroyAllWindows()


def save_icdar2015_rpn_image(proposal_list, img, img_metas, save_dir):
    if not os.path.exists(save_dir):               #判断文件夹是否存在
        os.makedirs(save_dir)                       #新建文件夹

    bbox_results = proposal_list[0].detach().cpu().numpy()
    polys = np.array(bbox_results).reshape(-1, 5)
    image_source = img.cpu().detach().numpy()
    image_source = np.transpose(image_source, (0, 2, 3, 1))
    mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1)
    std = np.array([58.395, 57.12, 57.375]).reshape(1, -1)
    image_source = np.multiply(image_source, std)
    image_source = np.add(image_source, mean).astype(np.uint8)
    image_source = image_source[0].copy()
    for id in range(polys.shape[0]):
        x1 = polys[id][0]
        y1 = polys[id][1]
        x2 = polys[id][2]
        y2 = polys[id][3]
        cv2.rectangle(image_source, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(save_dir, img_metas[0]['ori_filename']), image_source)


def save_icdar2015_rpn_box_txt(proposal_list, img_metas, save_dir):
    if not os.path.exists(save_dir):               #判断文件夹是否存在
        os.makedirs(save_dir)                       #新建文件夹

    bbox_results = proposal_list[0].detach().cpu().numpy()
    polys = np.array(bbox_results).reshape(-1, 5)
    image_jjj = img_metas[0]['ori_filename'][0:-4]
    scale_factor = img_metas[0]['scale_factor']
    with open('{}'.format(os.path.join(save_dir, 'res_{}.txt'.format(image_jjj))), 'w') as f:
        for id in range(polys.shape[0]):
            x1 = polys[id][0] / scale_factor[0]
            y1 = polys[id][1] / scale_factor[1]
            x2 = polys[id][2] / scale_factor[2]
            y2 = polys[id][3] / scale_factor[3]
            f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)))


def save_icdar2015_box_image(proposal_list, img, img_metas, save_dir):
    if not os.path.exists(save_dir):               #判断文件夹是否存在
        os.makedirs(save_dir)                       #新建文件夹

    polys = np.array(proposal_list).reshape(-1, 5)
    image_source = img.cpu().detach().numpy()
    image_source = np.transpose(image_source, (0, 2, 3, 1))
    mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1)
    std = np.array([58.395, 57.12, 57.375]).reshape(1, -1)
    image_source = np.multiply(image_source, std)
    image_source = np.add(image_source, mean).astype(np.uint8)
    image_source = image_source[0].copy()
    for id in range(polys.shape[0]):
        x1 = polys[id][0]
        y1 = polys[id][1]
        x2 = polys[id][2]
        y2 = polys[id][3]
        cv2.rectangle(image_source, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(save_dir, img_metas[0]['ori_filename']), image_source)


def save_icdar2015_box_txt(results, img_metas, save_dir):
    if not os.path.exists(save_dir):               #判断文件夹是否存在
        os.makedirs(save_dir)                       #新建文件夹

    polys = np.array(results[0]).reshape(-1, 5)
    image_jjj = img_metas[0]['ori_filename'][0:-4]
    with open('{}'.format(os.path.join(save_dir, 'res_{}.txt'.format(image_jjj))), 'w') as f:
        for id in range(polys.shape[0]):
            x1 = polys[id][0]
            y1 = polys[id][1]
            x2 = polys[id][2]
            y2 = polys[id][3]
            f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)))


def save_icdar2015_mask_box_image(proposal_list, img, img_metas, save_dir):
    if not os.path.exists(save_dir):               #判断文件夹是否存在
        os.makedirs(save_dir)                       #新建文件夹

    bbox_results = proposal_list[0].detach().cpu().numpy()
    polys = np.array(bbox_results).reshape(-1, 5)
    image_source = img.cpu().detach().numpy()
    image_source = np.transpose(image_source, (0, 2, 3, 1))
    mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1)
    std = np.array([58.395, 57.12, 57.375]).reshape(1, -1)
    image_source = np.multiply(image_source, std)
    image_source = np.add(image_source, mean).astype(np.uint8)
    image_source = image_source[0].copy()
    for id in range(polys.shape[0]):
        x1 = polys[id][0]
        y1 = polys[id][1]
        x2 = polys[id][2]
        y2 = polys[id][3]
        cv2.rectangle(image_source, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(save_dir, img_metas[0]['ori_filename']), image_source)


def save_icdar2015_mask_box_txt(point_box, img_metas, save_dir):
    if not os.path.exists(save_dir):               #判断文件夹是否存在
        os.makedirs(save_dir)                       #新建文件夹

    polys = np.array(point_box).reshape(-1, 8)
    image_name = img_metas['ori_filename'][0:-4]
    with open('{}'.format(os.path.join(save_dir, 'res_{}.txt'.format(image_name))), 'w') as f:
        for id in range(polys.shape[0]):
            f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                int(polys[id][0]), int(polys[id][1]), int(polys[id][2]), int(polys[id][3]),
                int(polys[id][4]), int(polys[id][5]), int(polys[id][6]), int(polys[id][7])))


#*****************************icdar2013***********************************
def save_icdar2013_rpn_box_txt(proposal_list, img_metas, save_dir):
    if not os.path.exists(save_dir):               #判断文件夹是否存在
        os.makedirs(save_dir)                       #新建文件夹

    bbox_results = proposal_list[0].detach().cpu().numpy()
    polys = np.array(bbox_results).reshape(-1, 5)
    image_jjj = img_metas[0]['ori_filename'][0:-4]
    scale_factor = img_metas[0]['scale_factor']
    with open('{}'.format(os.path.join(save_dir, 'res_{}.txt'.format(image_jjj))), 'w') as f:
        for id in range(polys.shape[0]):
            x1 = polys[id][0] / scale_factor[0]
            y1 = polys[id][1] / scale_factor[1]
            x2 = polys[id][2] / scale_factor[2]
            y2 = polys[id][3] / scale_factor[3]
            f.write('{}, {}, {}, {}\n'.format(
                int(x1), int(y1), int(x2), int(y2)))


def save_icdar2013_box_txt(results, img_metas, save_dir):
    if not os.path.exists(save_dir):               #判断文件夹是否存在
        os.makedirs(save_dir)                       #新建文件夹

    polys = np.array(results).reshape(-1, 5)
    image_jjj = img_metas[0]['ori_filename'][0:-4]
    with open('{}'.format(os.path.join(save_dir, 'res_{}.txt'.format(image_jjj))), 'w') as f:
        for id in range(polys.shape[0]):
            x1 = polys[id][0]
            y1 = polys[id][1]
            x2 = polys[id][2]
            y2 = polys[id][3]
            f.write('{}, {}, {}, {}\n'.format(
                int(x1), int(y1), int(x2), int(y2)))


def save_icdar2013_mask_box_txt(point_box, img_metas, save_dir):
    if not os.path.exists(save_dir):               #判断文件夹是否存在
        os.makedirs(save_dir)                       #新建文件夹

    polys = np.array(point_box).reshape(-1, 4)
    image_name = img_metas['ori_filename'][0:-4]
    with open('{}'.format(os.path.join(save_dir, 'res_{}.txt'.format(image_name))), 'w') as f:
        for id in range(polys.shape[0]):
            f.write('{}, {}, {}, {}\n'.format(
                int(polys[id][0]), int(polys[id][1]), int(polys[id][2]), int(polys[id][3])))


#*****************************ctw1500***********************************
def save_ctw1500_mask_box_txt(point_box, img_metas, save_dir):
    if not os.path.exists(save_dir):               #判断文件夹是否存在
        os.makedirs(save_dir)                       #新建文件夹

    image_name = img_metas['ori_filename'][0:-4]
    with open('{}'.format(os.path.join(save_dir, '{}.txt'.format(image_name))), 'w') as f:
        for id in range(len(point_box)):
            if len(point_box[id]) < 6:
                continue
            values = [int(v) for v in point_box[id]]
            for ii in range(len(values)):
                f.write('{}'.format(values[ii]))
                if ii < len(values) - 1:
                    f.write(', ')
            f.write('\n')

#**********************************************************************
def save_all_box_class(bbox, score, img_metas, save_dir):
    # save_dir = 'dataset/result/joint_regression'
    if not os.path.exists(save_dir):               #判断文件夹是否存在
        os.makedirs(save_dir)                       #新建文件夹
    image_name = img_metas[0]['ori_filename'][0:-4]
    with open('{}'.format(os.path.join(save_dir, '{}.txt'.format(image_name))), 'w') as f:
        assert len(bbox) == len(score)
        save_class_boxs =  torch.cat((bbox.int(), score[:, :-1]), 1)
        for id in range(len(save_class_boxs)):
            # values = [int(v) for v in save_class_boxs[id]]
            for ii in range(len(save_class_boxs[id])):
                f.write('{}'.format(save_class_boxs[id][ii]))
                if ii < len(save_class_boxs[id]) - 1:
                    f.write(', ')
            f.write('\n')