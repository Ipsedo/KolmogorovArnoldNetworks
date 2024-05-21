# -*- coding: utf-8 -*-
import argparse

from .options import ConvOptions, HermiteOptions, ModelOptions, TrainOptions
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser("kan main")

    _ = parser.parse_args()

    train(
        ModelOptions(
            ConvOptions(
                channels=[(3, 32), (32, 64), (64, 100)],
                kernel_sizes=[5, 5, 3],
                strides=[4, 4, 2],
                paddings=[2, 2, 1],
                residual_activation="mish",
            ),
            HermiteOptions(5),
        ),
        TrainOptions("./out/cifar100", 128, 1e-4, 1000, True),
    )


if __name__ == "__main__":
    main()
