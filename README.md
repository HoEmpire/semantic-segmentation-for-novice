# Repeat Fully Convolutional Network (FCN) by Pytoch
<div align=center>
<img width="" height="200" src="https://github.com/HoEmpire/semantic-segmentation-for-novice/raw/master/img/example.png"/>
</div>

## Table of Content
- [Background](#Background)
- [Requirement](#Requirement) 
- [Usage](#Usage) 
- [Reference](#Reference) 

## Background
- Repeat the FCN out of learning and researching purpose, especially for **autonomous driving scenario**.
- Due to the limitation of my GPU memory (2GB), I only implement a version of FCN with **resNet34**.
- I train the FCN with **CityScape Dataset**, the overall mean iou of fine label in CityScape Dataset is about **27%**, the [benchmark result](https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results) with resNet101 is **30.4%**, thus the performance is acceptable.
- It should be noted that the state-of-the-art performance of mean IoU is **84.5%**

## Requirement
- matplotlib==3.1.3
- numpy==1.18.1
- torch==1.3.1
- torchvision==0.4.2

## Usage
### step1: download the dataset
- Download the **CityScape Dataset** from [here](https://www.cityscapes-dataset.com/login/)
- You will have to create an account with your email address to download the dataset 
- Copy the dataset to ```./data```, the file sturcture of the workspace can be like the following:
<div align=center>
<img width="" height="" src="https://github.com/HoEmpire/semantic-segmentation-for-novice/raw/master/img/file structure.png"/>
</div>

### step2: train the FCN
- Setting the training parameter (like learning rate, weight decay, number of epoches, batch sizes, etc)
<div align=center>
<img width="" height="" src="https://github.com/HoEmpire/semantic-segmentation-for-novice/raw/master/img/parameter.png"/>
</div>

- Just run ```train.py```

### step3: evaluate the model
- Just run ```validation.py```

### step4: test the model on image from other dataset
- Then run ```test.py```
- Two examples from **RobotCar Dataset** are provided in ```./test```

## Reference
- Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 3431-3440.

