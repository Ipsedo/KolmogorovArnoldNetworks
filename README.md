# KolmogorovArnoldNetworks

Implementation of Kolmogorov-Arnold Networks with PyTorch.

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

Train on Cifar10

```bash
$ cd /path/to/KolmogorovArnoldNetworks
$ # with hermite
$ python -m kan "[(3,8),(8,16),(16,32),(32,64)]" "[(256,512),(512,10)]" -k 3 -s 2 -p 1 -r "silu" -a "hermite n=5" train "cifar10" "./out/cifar10" -lr 1e-4 -b 64 -e 100 --cuda
$ # with spline
$ python -m kan "[(3,8),(8,16),(16,32),(32,64)]" "[(256,512),(512,10)]" -k 3 -s 2 -p 1 -r "silu" -a "b-spline degree=2 grid_size=8" train "cifar10" "./out/cifar10" -lr 1e-4 -b 64 -e 100 --cuda
```

## Features

- Extension to 2D convolutions.
- Extension with Hermite polynomials, in progress !

## References

[1] KAN: Kolmogorov-Arnold Networks - *Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark* - 2 May 2024

[2] Wikipedia.org
