# DSP
Official implementation of "DSP: Dual Soft-Paste for Unsupervised Domain Adaptive Semantic Segmentation"

> Authors: Li Gao, Jing Zhang, Lefei Zhang, Dacheng Tao.

# Prerequisite

> - CUDA/CUDNN
> - Python3
> - PyTorch==1.7
> - Packages found in requirements.txt
1. Creat a new conda environment
```
conda create -n dsp_env python=3.7
conda install pytorch=1.7 torchvision torchaudio cudatoolkit -c pytorch
conda activate dsp_env
```
2. Download the code from github and change the directory

```
git clone https://github.com/GaoLii/DSP/
cd DSP
```
3. Prepare dataset

Download Cityscapes, GTA5 and SYNTHIA dataset, then organize the folder as follows:

```
├── ../../dataset/
│   ├── Cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── GTA5/
|   |   ├── images/
|   |   ├── labels/
│   ├── RAND_CITYSCAPES/ 
|   |   ├── GT/
|   |   ├── RGB/
...
```



# Training and Evaluation example

> Training and evaluation are on a single Tesla V100 GPU with 16G memory.

### Train with unsupervised domain adaptation 

#### GTA5->CityScapes 
```
python train_DSP.py --config ./configs/configUDA_gta.json --name UDA_gta
```
#### SYNTHIA->CityScapes
```
python train_DSP.py --config ./configs/configUDA_syn.json --name UDA_syn
```
### Evaluation 

```
python evaluateUDA.py --model-path checkpoint.pth
```


# Pretrained models
- [Pretrained model for GTA5->Cityscapes](Link: https://pan.baidu.com/s/10adjjSXarJOvat-ibzfoLg  Code: wv28): Peaked at 55.0 mIoU.

- [Pretrained model for SYN->Cityscapes]Link: https://pan.baidu.com/s/1cL7rYximzpUmYt-8OcpF4Q  Code: mw4e): Peaked at 63.8 mIoU.

This model should be unzipped in the '../saved' folder.

# License

The code is heavily borrowed from [DACS](https://github.com/vikolss/DACS).

If you use this code in your research please consider citing

```
@article    
```

