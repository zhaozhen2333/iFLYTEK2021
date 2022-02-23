import os
import sys
import json
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from PIL import Image
from skimage import io as skio

__places__ = [
    'CGDZ_9', 'CGGZ_7', 'CGHA_5', 'CGHE_2', 'CGHE_13', 'CGJC_23',
    'CGJC_24', 'CGJN_2', 'CGJQ_3', 'CGLF_9', 'CGLY_4', 'CGTL_25',
    'CGWH_4', 'CGWH_18', 'CGXC_3', 'CGZJ_6', 'CGZY_7', 'CGZY_24']


def save(images, annotations):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations

    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    ann['categories'] = categories
    json.dump(ann, open(poi_json_path, 'w'))


def test_dataset(im_lists):
    """
    对于没有ground truth的数据集，使用 mmdetection 框架生成预测结果，需要生成一个伪标签文件test.json
    :param im_lists: list[],包含所有原始places图片剪切成小图后，每张小图的路径
    :return: 保存伪标签文件test.json，具有正确的key和随意设定的value
    """
    idx = 0
    image_id = 0
    images = []
    annotations = []
    for im_path in tqdm(im_lists):
        im = Image.open(im_path)
        w, h = im.size
        image = {'file_name': os.path.basename(im_path), 'width': w, 'height': h, 'id': image_id}
        images.append(image)

        labels = [[10, 10, 20, 20]]
        for label in labels:
            bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
            seg = []
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': 0, 'id': idx, 'ignore': 0}

            annotations.append(ann)
            idx += 1
        image_id += 1
    save(images, annotations)


def lonlat2imagexy(dataset, lon, lat):
    """
    根据地理坐标(经纬度)转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param lon: 经度坐标
    :param lat: 纬度坐标
    :return: 地理坐标(lon,lat)对应的影像图上行列号(row, col)
    """
    transform = dataset.GetGeoTransform()
    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]
    x_pix = (lon - x_origin) / pixel_width
    y_pix = (lat - y_origin) / pixel_height
    return x_pix, y_pix


def Clip_box_lists(img_width, img_height):
    """
    在原始图片上生成滑窗，这些滑窗用于剪切出一系列小图以组成数据集
    选择每张原始place图片的左上角的点作为零点生成坐标系，图片的宽为X轴，高为Y轴。
    每个滑窗的左上角坐标为(x0, y0)，右下角坐标为(x1, y1)，
    滑窗表示为[y0, y1, x0, x1]
    :param img_width: 原始图片的宽
    :param img_height: 原始图片的高
    :return: list[list[y0, y1, x0, x1],...]
    """
    clip_box_lists = []
    w_num = 0
    h, w = [scale, scale]  # 小图的高与宽
    while w_num < img_width:
        left_top_width = w_num
        h_num = 0

        if w_num + w < img_width:
            right_sub_width = w_num + w
            while h_num < img_height:
                if h_num + h < img_height:
                    left_top_height = h_num
                    right_sub_height = h_num + h
                    clip_box_list = [left_top_height, right_sub_height, left_top_width, right_sub_width]
                    h_num += stride  # 滑窗首先向下平移stride长度
                    clip_box_lists.append(clip_box_list)
                else:
                    left_top_height = h_num  # 当滑窗下边界超过原始图片的高度时，停止移动，并取原始图片的下边界为滑窗下边界
                    right_sub_height = img_height
                    clip_box_list = [left_top_height, right_sub_height, left_top_width, right_sub_width]
                    clip_box_lists.append(clip_box_list)
                    break
            w_num += stride  # 滑窗向右平移stride长度
        else:
            right_sub_width = img_width  # 当滑窗右边界超过原始图片的宽度时，停止移动，并取原始图片的右边界为滑窗右边界
            while h_num < img_height:
                if h_num + h < img_height:
                    left_top_height = h_num
                    right_sub_height = h_num + h
                    clip_box_list = [left_top_height, right_sub_height, left_top_width, right_sub_width]
                    h_num += stride
                    clip_box_lists.append(clip_box_list)
                else:
                    left_top_height = h_num  # 当滑窗右、下边界同时超过原始图片时，取原始图片的右、下边界为滑窗右、下边界
                    right_sub_height = img_height
                    clip_box_list = [left_top_height, right_sub_height, left_top_width, right_sub_width]
                    clip_box_lists.append(clip_box_list)
                    break
            break
    return clip_box_lists


def data_process(tif_path, img_path):
    """
    使用生成的滑窗在原始图片上剪切出一系列小图并保存
    :param tif_path: 原始tif数据的路径
    :param img_path: 小图的保存路径
    :return: list[]，包含原始图片剪切成小图后，每张小图的路径
    """
    # 1 读取栅格数据
    dataset = gdal.Open(tif_path)
    img_width = dataset.RasterXSize  # 栅格矩阵的列数
    img_height = dataset.RasterYSize  # 栅格矩阵的行数
    img_bands = dataset.RasterCount  # 波段数
    img_data_type = gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)  # 原始数据类型

    # 3 设定影像分块尺寸
    clip_box_lists = Clip_box_lists(img_width, img_height)
    im_lists = []
    # 4 根据滑窗对相应的目标框进行处理
    for i_clip, clip_box in enumerate(clip_box_lists):
        xmin = clip_box[2]
        ymin = clip_box[0]
        width = (clip_box[3] - clip_box[2])
        height = (clip_box[1] - clip_box[0])
        # 存储图片
        # out_img_path = osp.join(tif_path,f"{i_clip}.jpg")
        out_img_path = img_path + f"_{i_clip}.jpg"
        im_lists.append(out_img_path)
        img_data_int8 = dataset.ReadAsArray(xmin, ymin, width, height).astype(np.uint8)  # 获取分块数据
        img_data = np.transpose(img_data_int8, (1, 2, 0))[:, :, [2, 1, 0]]
        skio.imsave(out_img_path, img_data)
    return im_lists


if __name__ == '__main__':
    # 1 读取滑窗的尺寸scale，移动的步长stride
    data_root = sys.argv[1]
    scale, stride = map(int, data_root.split('-'))
    
    # 2 读取路径与创建文件夹
    label_ids = {"CultivatedLand": 0}
    test_img_path = 'out_shp/inference/images/'  # tif文件所在的文件夹
    test_clip_path = f'out_shp/inference/{data_root}/test/'  # 剪切后的小图所在的文件夹
    poi_json_path = f'out_shp/inference/{data_root}/test.json'  # 伪标签test.json的路径
    os.makedirs(test_clip_path, exist_ok=True)
    
    # 3 对于每张place的原始图片，保存剪切得到的小图
    if sys.argv[3] == 'no':
        place = sys.argv[2]
        tif_path = test_img_path + f"{place}_offset.tif"
        img_path = test_clip_path + f"{place}_offset"
        im_lists = data_process(tif_path, img_path)
        #im_lists.sort(key=lambda x: int((x.split('_')[-1]).split('.')[0]))

        os.makedirs('tmp', exist_ok=True)
        with open(f"tmp/{place}.txt", 'w') as f:
            f.write("\n".join(im_lists))  # 创建文件夹tmp,存储每个place剪切得到的小图的路径
    
    # 4 所有原始place图片剪切完成后，利用保存的小图路径信息的txt文件创建伪标签文件test.json
    else:
        import time

        starttime = time.time()

        os.makedirs('tmp', exist_ok=True)
        while len(os.listdir('tmp')) < 18:
            pass

        time.sleep(2)

        multi_im_lists = []  # 读取所有place剪切得到的小图的路径，创建伪标签文件
        for place in __places__:
            im_lists = open(f'tmp/{place}.txt').read().rstrip().split('\n')
            multi_im_lists.extend(im_lists)

        test_dataset(multi_im_lists)

        endtime = time.time()
        print(place, " runtime:", endtime - starttime)

