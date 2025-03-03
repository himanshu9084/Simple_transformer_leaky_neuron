# Simple Transformer with Single Leaky Neuron for Event Vision

Code for the paper: **Simple Transformer with Single Leaky Neuron for Event Vision**, accepted at EVGEN: Event-based Vision in the Era of Generative AI - Transforming Perception and Visual Innovation Workshop, WACV 2025.

---

## Table of Contents
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Executing Program](#executing-program)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Installation

1. Install [spikingjelly](https://github.com/fangwei123456/spikingjelly) from Github only.
2. Install rest of the packages as mentioned in requirements.txt .

## Dataset Setup

1. Download the datasets manually from the [spikingjelly datasets repository](https://github.com/fangwei123456/spikingjelly/tree/master/spikingjelly/datasets):
   - DVS Gesture
   - N-MNIST
   - CIFAR10-DVS
2. Place the downloaded datasets in the `event_vision/datasets/` folder. The code will automatically preprocess the datasets into frame-based versions during execution.

## Executing Program

* Download pretrained Resnet18 and Resnet50 weights from pytorch.
* Update the dataset paths in the `custom_train.py` file to match your setup.
* To start training, run the following command:
   ```bash
   python custom_train.py
   ```
* Logs and training progress will be stored in the `logs/` folder.

## Citation

If you find this code useful in your research, please consider citing our paper:
```
@InProceedings{Kumar_2025_WACV,
    author    = {Kumar, Himanshu and Konkar, Aniket},
    title     = {Simple Transformer with Single Leaky Neuron for Event Vision},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {February},
    year      = {2025},
    pages     = {928-934}
}
```

---

## Acknowledgements

We have used [Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron](https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/) code, and [syops-counter](https://github.com/iCGY96/syops-counter) for energy calculation.
