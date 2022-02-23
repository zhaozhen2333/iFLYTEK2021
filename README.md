# The-iFLYTEK-2021-Cultivatedland-Extraction-From-High-Resolution-Remote-Sensing-Image-Challenge
The Winning Solution to the Cultivated land Extraction From High-Resolution Remote Sensing Image Challenge
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
