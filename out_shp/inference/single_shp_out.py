import os
import os.path as osp
import sys
import cv2
import time
import json
import torch
import numpy as np
from osgeo import gdal, ogr, osr
from pandas import DataFrame
from skimage import measure
from shapely.geometry import Polygon
from pycocotools import mask as mask_util
from multiprocessing.pool import Pool


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
                    clip_box_list = [
                        left_top_height, right_sub_height, left_top_width,
                        right_sub_width
                    ]
                    h_num += stride  # 滑窗首先向下平移stride长度
                    clip_box_lists.append(clip_box_list)
                else:
                    left_top_height = h_num  # 当滑窗下边界超过原始图片的高度时，停止移动，并取原始图片的下边界为滑窗下边界
                    right_sub_height = img_height
                    clip_box_list = [
                        left_top_height, right_sub_height, left_top_width,
                        right_sub_width
                    ]
                    clip_box_lists.append(clip_box_list)
                    break
            w_num += stride  # 滑窗向右平移stride长度
        else:
            right_sub_width = img_width  # 当滑窗右边界超过原始图片的宽度时，停止移动，并取原始图片的右边界为滑窗右边界
            while h_num < img_height:
                if h_num + h < img_height:
                    left_top_height = h_num
                    right_sub_height = h_num + h
                    clip_box_list = [
                        left_top_height, right_sub_height, left_top_width,
                        right_sub_width
                    ]
                    h_num += stride
                    clip_box_lists.append(clip_box_list)
                else:
                    left_top_height = h_num  # 当滑窗右、下边界同时超过原始图片时，取原始图片的右、下边界为滑窗右、下边界
                    right_sub_height = img_height
                    clip_box_list = [
                        left_top_height, right_sub_height, left_top_width,
                        right_sub_width
                    ]
                    clip_box_lists.append(clip_box_list)
                    break
            break
    return clip_box_lists


def data_process(tif_single_path, img_path, json_path):
    """
    保存每张小图的左上角点的坐标，以便合并结果
    :param tif_single_path: 原始tif数据的路径
    :param img_path: 小图的保存路径
    :param json_path: 包含小图坐标信息的json文件
    """
    # 读取栅格数据
    dataset = gdal.Open(tif_single_path)
    img_width = dataset.RasterXSize     # 栅格矩阵的列数
    img_height = dataset.RasterYSize    # 栅格矩阵的行数
    clip_box_lists = Clip_box_lists(img_width, img_height)

    # 根据裁剪框对相应的目标框进行处理
    json_dict = []
    for i_clip, clip_box in enumerate(clip_box_lists):
        xmin = clip_box[2]
        ymin = clip_box[0]
        width = (clip_box[3] - clip_box[2])
        height = (clip_box[1] - clip_box[0])
        # 图片路径
        out_img_path = img_path + f"_{i_clip}.jpg"
        # 存储坐标json
        out_json_path = json_path + f".json"
        part_json_dict = {
            "coordinate": [xmin, ymin],
            "imagePath": out_img_path,
            "imageHeight": height,
            "imageWidth": width
        }

        json_dict.append(part_json_dict)
        with open(out_json_path, 'w') as output_json_file:
            json.dump(json_dict, output_json_file, indent=4)


def id_to_filename(path):
    """
    建立剪切图片的 id 与 file_name 的转换字典
    数据集为coco数据集格式，利用 mmdetection 框架进行预测时得到的预测文件只包含图片的id信息
    :param path: json的路径
    :return: dict{id : file_name}
    """
    test_jsons = json.load(open(path))
    images = test_jsons['images']
    Frame = DataFrame(images)
    file_name = np.array(Frame['file_name']).tolist()
    id = np.array(Frame['id']).tolist()
    images_dict = dict(zip(id, file_name))
    return images_dict


def arrange_json(place, coordinate_json, out_segm_json, out_augtest_json):
    """
    将 mmdetection 的预测结果分别保存到对应的原始place图片的place.json中；
    为了便于矩阵计算，将place_offset_json维度扩充
    :param place: 图片的地名
    :param coordinate_json: 包含小图片坐标信息的json文件
    :param out_segm_json: 单个place原始图片的预测结果
    :param out_augtest_json: 扩充后的aug_place_offset_coord_json
    :return:
    """
    # 1 读取 mmdetection 的预测文件segm.json
    segm_jsons = json.load(open(segm_result))

    # 2 读取伪标签test.json，得到id to file_name的转换字典
    images_dict = id_to_filename(poi_json_path)

    # 3 读取坐标文件place_offset_coord.json，得到file_name to coordinate的转换字典
    coordinate_jsons = json.load(open(coordinate_json))
    Frame = DataFrame(coordinate_jsons)
    imagePath = np.array(Frame['imagePath']).tolist()
    imagePath_dict = dict(zip(imagePath, coordinate_jsons))

    # 4 将预测文件分别按place,划分到不同的place.json; 扩充坐标文件与place.json的长度相同得到aug_place_offset_coord.json
    list_CGDZ_8 = []
    aug_CGDZ_8 = []
    for segm_json in segm_jsons:
        image_id = segm_json['image_id']
        file_name = images_dict[image_id]
        if file_name.startswith(place):
            list_CGDZ_8.append(segm_json)
            aug_CGDZ_8.append(imagePath_dict[file_name])
    with open(out_segm_json, 'w') as output_json_file_0:
        json.dump(list_CGDZ_8, output_json_file_0, indent=4)
    with open(out_augtest_json, 'w') as output_json_file_1:
        json.dump(aug_CGDZ_8, output_json_file_1, indent=4)


def arrange(medium_path, init_path, out_json):
    """
    运行边界筛选法筛选满足要求的预测结果
    :param medium_path: 每张原始place图片的预测结果place.json
    :param init_path: 扩充后的坐标文件aug_place_offset_coord.json
    :param out_json: 运用边界筛选法筛选后的预测结果select_place.json

    [score, imageHeight, imageWidth, xmin, ymin, bboxx, bboxy]
    score： tensor(N,1)，预测mask结果的置信度
    imageHeight, imageWidth： tensor(N,1)，小图的高和宽
    xmin, ymin： tensor(N,1)，小图左上角顶点的坐标
    bboxx, bboxy： tensor(N,1)，预测结果的bbox左上角顶点的坐标
    """
    # 1 读取预测文件place.json
    segm_jsons = json.load(open(medium_path))
    segm_Frame = DataFrame(segm_jsons)
    score = torch.tensor(np.array(segm_Frame[['score']]).tolist())  # score: tensor(N,1)
    bbox = torch.tensor(np.array(segm_Frame[['bbox']]).tolist())  # bbox: tensor(N,1,4)
    bboxx = bbox[:, :, 0]  # bboxx: tensor(N,1)
    bboxy = bbox[:, :, 1]  # bboxy: tensor(N,1)

    # 2 读取扩充后的坐标文件aug_place_offset_coord.json
    inits_json = json.load(open(init_path))
    Frame = DataFrame(inits_json)
    coordinate = torch.tensor(np.array(Frame[['coordinate']]).tolist()).float()  # tensor(N,1,2)
    imageHeight = torch.tensor(np.array(Frame[['imageHeight']]).tolist()).float()  # tensor(N,1)
    imageWidth = torch.tensor(np.array(Frame[['imageWidth']]).tolist()).float()  # tensor(N,1)
    xmin = coordinate[:, :, 0]  # tensor(N,1)
    xmin = xmin - min(xmin)
    ymin = coordinate[:, :, 1]  # tensor(N,1)

    # 3 将数据cat为矩阵形式进行计算，以提高计算速度
    judge_condition = torch.cat(
        [score, imageHeight, imageWidth, xmin, ymin, bboxx, bboxy], dim=1)  # tensor(N,6)

    # 4 运用边界筛选法
    keep_mask_0 = (judge_condition[:, 0] > score_thr)  # 对score进行筛选
    keep_inds_0 = torch.nonzero(keep_mask_0, as_tuple=False).reshape(-1)
    judge_condition_1 = judge_condition[keep_mask_0]
    segm_jsons_1 = [segm_jsons[idx] for idx in keep_inds_0]

    keep_mask_1 = (judge_condition_1[:, 1] == scale) & (judge_condition_1[:, 2] == scale) & \
                  (((judge_condition_1[:, 3] != 0) & (judge_condition_1[:, 4] != 0) &
                    (2 <= judge_condition_1[:, 5]) & (judge_condition_1[:, 5] <= stride) &
                    (2 <= judge_condition_1[:, 6]) & (judge_condition_1[:, 6] <= stride)) |
                   ((judge_condition_1[:, 3] != 0) & (judge_condition_1[:, 4] == 0) &
                    (2 <= judge_condition_1[:, 5]) & (0 <= judge_condition_1[:, 6]) &
                    (judge_condition_1[:, 5] <= stride) & (judge_condition_1[:, 6] <= stride)) |
                   ((judge_condition_1[:, 3] == 0) & (judge_condition_1[:, 4] != 0) &
                    (0 <= judge_condition_1[:, 5]) & (2 <= judge_condition_1[:, 6]) &
                    (judge_condition_1[:, 5] <= stride) & (judge_condition_1[:, 6] <= stride)) |
                   ((judge_condition_1[:, 3] == 0) & (judge_condition_1[:, 4] == 0) &
                    (0 <= judge_condition_1[:, 5]) & (0 <= judge_condition_1[:, 6]) &
                    (judge_condition_1[:, 5] <= stride) & (judge_condition_1[:, 6] <= stride)))
    # 默认原始place图片下边界剪切得到的小图的高imageHeight小于标准高scale
    keep_mask_2 = (judge_condition_1[:, 1] < scale) & (judge_condition_1[:, 2] == scale) & \
                  (((judge_condition_1[:, 3] == 0) &
                    (0 <= judge_condition_1[:, 5]) & (2 <= judge_condition_1[:, 6]) &
                    (judge_condition_1[:, 5] <= stride)) |
                   ((judge_condition_1[:, 3] != 0) &
                    (2 <= judge_condition_1[:, 5]) & (2 <= judge_condition_1[:, 6]) &
                    (judge_condition_1[:, 5] <= stride)))
    # 默认原始place图片右边界剪切得到的小图的宽imageWidth小于标准宽scale
    keep_mask_3 = (judge_condition_1[:, 1] == scale) & (judge_condition_1[:, 2] < scale) & \
                  (((judge_condition_1[:, 4] == 0) &
                    (0 <= judge_condition_1[:, 6]) & (2 <= judge_condition_1[:, 5]) &
                    (judge_condition_1[:, 6] <= stride)) |
                   ((judge_condition_1[:, 4] != 0) &
                    (2 <= judge_condition_1[:, 6]) & (2 <= judge_condition_1[:, 5]) &
                    (judge_condition_1[:, 6] <= stride)))
    # 则原始place图片剪切得到的右下角位置的最后一张小图的高和宽同时小于标准高和宽scale
    keep_mask_4 = (judge_condition_1[:, 1] < scale) & (judge_condition_1[:, 2] < scale) & \
                  ((2 <= judge_condition_1[:, 6]) & (2 <= judge_condition_1[:, 5]))

    # 5 合并筛选结果，并保存结果到select_place.json
    keep_mask = keep_mask_1 | keep_mask_2 | keep_mask_3 | keep_mask_4

    keep_inds = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
    filtered_segm_jsons = [segm_jsons_1[idx] for idx in keep_inds]

    with open(out_json, 'w') as output_json_file:
        json.dump(filtered_segm_jsons, output_json_file, indent=4)


def close_contour(contour):
    """
    闭合提取到的边界
    """
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask):
    """
    提取预测mask的边界
    """
    areas = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours, _ = cv2.findContours(
        padded_binary_mask,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)

    assert len(areas) == len(contours)
    if len(areas) > 0:
        max_index = areas.index(max(areas))
        contour = np.squeeze(contours[max_index], axis=1)
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance=0)
        polygons = contour.tolist()
    else:
        polygons = []

    return polygons


class GDAL_shp_Data(object):

    def __init__(self, shp_single_path):
        self.shp_single_path = shp_single_path
        self.shp_file_create()

    def shp_file_create(self):
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
        ogr.RegisterAll()
        driver = ogr.GetDriverByName("ESRI Shapefile")
        # 打开输出文件及图层
        # 输出模板shp 包含待写入的字段信息
        self.outds = driver.CreateDataSource(self.shp_single_path)
        # 创建空间参考
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        # 创建图层
        self.out_layer = self.outds.CreateLayer("out_polygon", srs,
                                                ogr.wkbPolygon)
        field_name = ogr.FieldDefn("scores", ogr.OFTReal)
        self.out_layer.CreateField(field_name)

    def set_shapefile_data(self, polygons, scores):
        for i in range(len(scores)):
            wkt = polygons[i].wkt  # 创建wkt文本点
            temp_geom = ogr.CreateGeometryFromWkt(wkt)
            feature = ogr.Feature(self.out_layer.GetLayerDefn())  # 创建特征
            feature.SetField("scores", scores[i])
            feature.SetGeometry(temp_geom)
            self.out_layer.CreateFeature(feature)
        self.finish_io()

    def finish_io(self):
        del self.outds


def xy2lonlat(points, tif_single_path):
    """
    图像坐标转经纬度坐标
    """
    dataset = gdal.Open(tif_single_path)
    transform = dataset.GetGeoTransform()
    lonlats = np.zeros(points.shape)

    lonlats[:, 0] = transform[
        0] + points[:, 0] * transform[1] + points[:, 1] * transform[2]
    lonlats[:, 1] = transform[
        3] + points[:, 0] * transform[4] + points[:, 1] * transform[5]
    return lonlats


def coordinate_change(ini_single_path, index):
    """
    由图片的id得到图片的coordinate的转换函数
    :param ini_single_path: 图片的坐标文件place_offset_coord.json
    :param index: 图片在coco数据集上的索引 id
    :return: 图片左上角点的坐标 xmin, ymin
    """
    # 1 读取伪标签test.json，得到id to file_name的转换字典
    images_dict = id_to_filename(poi_json_path)
    
    # 2 读取坐标文件place_offset_coord.json，得到file_name to coordinate的转换字典
    inits_json = json.load(open(ini_single_path))
    Frame = DataFrame(inits_json)
    imagePath = np.array(Frame['imagePath']).tolist()
    coordinate = np.array(Frame['coordinate']).tolist()
    coordinates_dict = dict(zip(imagePath, coordinate))
    
    # 3 得到id to coordinate的转换函数
    file_name2 = images_dict[index]
    coordinate2 = coordinates_dict[file_name2]
    return coordinate2[0], coordinate2[1]


def polygen_data_func(args):
    """
    读取预测结果的文件select_place.json，转换为输出数据polygon类型和score；
    由坐标文件place_offset_coord.json，将输出结果还原回原始place图片中
    Args:
        segm_json: 模型预测结果的文件select_place.json
        ini_single_path: 图片的坐标文件place_offset_coord.json
    """
    segm_json, ini_single_path = args
    segmentation = segm_json['segmentation']
    score = segm_json['score']
    image_id = segm_json['image_id']
    xmin, ymin = coordinate_change(ini_single_path, image_id)  # 每个id对应的小图的坐标
    binary_mask = mask_util.decode(segmentation)  # segmentation类型转换为binary mask类型
    polygons = binary_mask_to_polygon(binary_mask)  # binary mask类型提取出边界

    polygon_lonlat = None
    if len(polygons) > 3:
        for i in range(len(polygons)):
            a, b = polygons[i]
            polygons[i] = [a + xmin, b + ymin]  # 每个polygons坐标还原回原位置
        polygon_data_np = np.array(polygons, dtype='object')
        polygon_lonlat = Polygon(xy2lonlat(polygon_data_np, tif_single_path))  # 数据转换为Polygon类型的数据
    return polygon_lonlat, score


def polygen_data(medium_path, ini_single_path):
    """
    :param medium_path: 每张原始place图片筛选后的预测结果select_place.json
    :param ini_single_path: 坐标文件place_offset_coord.json
    :return: list_CGDZ_8: list[...],每张原始place图片的Polygons; 
            scores_CGDZ_8: list[...],每张原始place图片的scores
    """
    segm_jsons = json.load(open(medium_path))

    pool = Pool(processes=3)
    args = [(segm_json, ini_single_path) for segm_json in segm_jsons]
    results = pool.map(polygen_data_func, args)

    list_CGDZ_8 = []
    scores_CGDZ_8 = []
    for polygon_lonlat, score in results:
        if polygon_lonlat is not None:
            list_CGDZ_8.append(polygon_lonlat)
            scores_CGDZ_8.append(score)

    return list_CGDZ_8, scores_CGDZ_8


def out_shp(place):
    """
    输出为.shp文件
    :param place: 每张原始place数据的名称
    :return: 每张原始place数据的耕地目标提取文件.shp
    """
    medium_path = ini_poi_path + f"select_{place}.json"  # 每张原始place图片筛选后的预测结果select_place.json
    ini_single_path = ini_poi_path + f"{place}_offset_coord.json"  # 坐标文件place_offset_coord.json
    polygon_lonlat_list, scores_list = polygen_data(medium_path, ini_single_path)
    shp_single_path = osp.join(shp_path, f"{place}_offset.shp")  # .shp文件输出路径
    shp_data = GDAL_shp_Data(shp_single_path)
    shp_data.set_shapefile_data(polygon_lonlat_list, scores_list)  # 输出为符合要求的.shp文件


if __name__ == '__main__':
    data_root = sys.argv[1]
    scale, stride = map(int, data_root.split('-'))
    score_thr = 0.004

    tif_path = 'out_shp/inference/images/'
    segm_result = f'out_shp/inference/{data_root}.segm.json'
    shp_path = f'out_shp/inference/{data_root}/out_shp'
    ini_poi_path = f'out_shp/inference/{data_root}/inipoi/'
    poi_json_path = f'out_shp/inference/{data_root}/test.json'

    os.makedirs(shp_path, exist_ok=True)
    os.makedirs(ini_poi_path, exist_ok=True)

    starttime = time.time()

    # 1 保存每张小图的左上角点的坐标，以便合并结果
    place = sys.argv[2]
    tif_single_path = tif_path + f"{place}_offset.tif"
    img_single_path = f"{place}_offset"
    coordinate_json = ini_poi_path + f"{place}_offset_coord"
    data_process(tif_single_path, img_single_path, coordinate_json)

    # 2 对 mmdetection 得到的segm.json文件进行筛选，得到符合要求的select_place.json
    # 将总体的预测结果segm.json拆分到对应的place.json中；为了便于计算，将坐标文件扩充为aug_coord_offset_coord_json
    segm_json = ini_poi_path + f'{place}.json'
    out_augtest_json = ini_poi_path + f'aug_{place}_offset_coord.json'
    arrange_json(place, coordinate_json + ".json", segm_json, out_augtest_json)
    # 运行边界筛选法筛选满足要求的预测结果select_place.json
    out_json = ini_poi_path + f"select_{place}.json"
    arrange(segm_json, out_augtest_json, out_json)

    # 3 将预测结果select_place.json转换为Polygon类型的数据，并还原到原位置，输出结果
    out_shp(place)

    endtime = time.time()
    print(place, " runtime:", endtime - starttime)
