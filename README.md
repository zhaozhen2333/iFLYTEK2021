# The-iFLYTEK-2021-Cultivatedland-Extraction-From-High-Resolution-Remote-Sensing-Image-Challenge
Our code was developed based on the mmdetection
## 0. Dataset
Thanks iFLYTEK and ChangGuang Satellite
All images and their associated annotations in DOTA can be used for academic purposes only, but any commercial use is prohibited.
dataset are available at:
链接：https://pan.baidu.com/s/1_yFbJ6nX1ovOK0_9BZ5Lrg?pwd=1234 
提取码：1234

## 1. 环境配置

本地运行的环境配置是针对linux系统和2080Ti显卡，如果测试时，遇到环境配置不符合，还请再联系

- **1. pytorch安装**

  下载anaconda

  ``` shell
  wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-Linux-x86_64.sh
   ```
  安装anaconda

  ``` shell
  chmod +x Anaconda3-2020.11-Linux-x86_64.sh
  ./Anaconda3-2020.11-Linux-x86_64.sh
   ```
  创建虚拟环境torch_1_7

  ``` shell
  conda create -n torch_1_7 python=3.7
   ```
  进入虚拟环境torch_1_7

  ``` shell
  conda activate torch_1_7
   ```
  安装pytorch

  ``` shell
  conda install pytorch=1.7.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
   ```
  3080Ti显卡需要CUDA11.0及以上，安装pytorch版本如下

  ``` shell
  conda install pytorch=1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch
   ```
- **2. mmdetection安装**
 
  安装MMDetection和MIM，它会自动处理OpenMMLab项目的依赖，包括mmcv等python包 
  
  ``` shell
  pip install openmim
  mim install mmdet
   ```
  可能出现找不到dist_train.sh和dist_test.sh的情况，请先运行
  
  ``` shell
  cd mmdetection
  chmod 777 ./tools/dist_train.sh
  chmod 777 ./tools/dist_test.sh
   ```
  MMDetection 是一个基于 PyTorch 的目标检测开源工具箱。它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。
  MMDetection安装文档：[快速入门文档](docs/get_started.md)

- **3. 必须的函数包安装**
  
  安装sklearn
  ``` shell
  pip install sklearn
   ```
  安装imgaug
  ``` shell
  pip install imgaug
   ```
  安装shapefile
  ``` shell
  pip install pyshp
   ```
  安装tqdm
  ``` shell
  pip install tqdm
   ```
  安装gdal
  ``` shell
  conda install -c conda-forge gdal
   ```
  安装shapely
  ``` shell
  pip install shapely
   ```
  安装skimage
  ``` shell
  pip install scikit-image
   ```
## 2. 运行说明
- **文件说明**

  文件结果如图所示,请您在训练前将初赛的原始数据复制到train/init_images内，tif文件直接放在image文件夹下，shp文件直接放在label文件夹下；\
  测试前将复赛数据tif文件直接复制到inference/images内
