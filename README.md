# KolmogorovArnoldNetworks

Implementation of Kolmogorov-Arnold Networks with PyTorch.

What is new in this project :
- Extension to 2D convolutions.
- Extension to Hermite polynomials, in progress !

## Installation

```bash
$ cd /path/to/KolmogorovArnoldNetworks
$ # install venv
$ python -m venv venv
$ source ./venv/bin/activate
$ # install requirements
$ pip install -r requirements.txt
$ # install pre-commit (needs "git init" command)
$ pre-commit install
```

## Run example

Train on Cifar10 :

```bash
$ cd /path/to/KolmogorovArnoldNetworks
$ # with hermite of degree 5
$ python -m kan "[(3,8),(8,16),(16,32),(32,64)]" "[(256,512),(512,10)]" -k 3 -s 2 -p 1 -r "silu" -a "hermite" -a "n=5" train "cifar10" "./out/cifar10" "./out/train_hermite_cifar10" -lr 1e-4 -b 64 -e 100 --train-ratio 0.7 --cuda --save-every 256
$ # with spline of degree 2 and grid size 8
$ python -m kan "[(3,8),(8,16),(16,32),(32,64)]" "[(256,512),(512,10)]" -k 3 -s 2 -p 1 -r "silu" -a "b-spline" -a "degree=2" -a "grid_size=8" train "cifar10" "./out/cifar10" "./out/train_bspline_cifar10" -lr 1e-4 -b 64 -e 100 --train-ratio 0.7 --cuda --save-every 256
```

Train on ImageNet :

```bash
$ # 1. download data on kaggle : https://www.kaggle.com/c/imagenet-object-localization-challenge
$ # 2. remember about the path of ImageNet dataset
$ # 3. start training :
$ cd /path/to/KolmogorovArnoldNetworks
$ # small network with hermite of degree 5
$ python -m kan "[(3,8),(8,16),(16,32),(32,64)]" "[(256,1000)]" -r "silu" -k 5 -s 4 -p 2 -a "hermite" -ao "n=5" --cuda train imagenet "/path/to/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train" "./out/train_imagenet" -e 1000 -b 64 -lr 1e-3 --save-every 4096 --train-ratio 0.7
```

## References

[1] KAN: Kolmogorov-Arnold Networks - *Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark* - 2 May 2024

[2] Wikipedia.org
