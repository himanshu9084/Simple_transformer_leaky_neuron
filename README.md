# Simple Transformer with Single Leaky Neuron for Event Vision

Code for the paper : **Simple Transformer with Single Leaky Neuron for Event Vision** accepted in EVGEN : Event-based Vision in the Era of Generative AI - Transforming Perception and Visual Innovation Workshop, WACV 2025.


## Installation

* Install [spikingjelly](https://github.com/fangwei123456/spikingjelly) from Github only.
* Install rest of the packages as mentioned in requirements.txt .

## Dataset Setup
* Download DVS Gesture, N-MNIST, CIFAR10-DVS dataset from spiking jelly library manually. Download links are available at [repo](https://github.com/fangwei123456/spikingjelly/tree/master/spikingjelly/datasets).
* Setup datasets in event_vision/datasets/ folder. The code will automatically create frame based version of datasets.

## Executing program

* Download pretrained Resnet18 and Resnet50 weights from pytorch.
* Set data paths in custom_train.py file.
* For training
```
python custom_train.py
```
* Logs will be generated in logs/ folder.

## EVGen paper
If you find this code useful in your research, please consider citing:

## Acknowledgements

We have used [Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron](https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/) code, and [syops-counter](https://github.com/iCGY96/syops-counter) for energy calculation.
