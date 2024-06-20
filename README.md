# DL2024_HW2_DCM
## setup
> git clone git@github.com:shangyuan191/DL2024_HW2_DCM.git

> conda create -n DLHW2 python=3.11

> conda activate DLHW2

> pip install -r requirements.txt

## environment
* CUDA 11.8
* pytorch 2.3.0

## dataset
* URL:https://cchsu.info/files/images.zip

## run
> python shang_main.py

## description
* #### task1:
> Design a special convolutional module that is spatial size invariant and can handle an arbitrary number of input channels. You only need to design this special module, not every layer of the CNN. After designing, explain the design principles, references, additional costs (such as FLOPS or #PARAMS), and compare with naive models. To simulate the practicality of your method, use the ImageNet-mini dataset for training, and during the inference process, test images with various channel combinations (such as RGB, RG, GB, R, G, B, etc.) and compare the performance.

* #### task2:
> Design a (2-4)-layer CNN, Transformer, or RNN network that can achieve 90% performance of ResNet34 on ImageNet-mini (i.e., with no more than 10% performance loss). There are no restrictions on parameter count or FLOPS, but the maximum number of input and output layers is limited to 4-6. Explain the design principles, references, and provide experimental results. We suggest you DO NOT use pre-trained models for ResNet34.

### Please refer to the report for more details.


