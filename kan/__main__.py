# -*- coding: utf-8 -*-
import argparse

from .train import HermiteConv2dKanOptions, TrainOptions, train


def main() -> None:
    parser = argparse.ArgumentParser("kan main")

    _ = parser.parse_args()

    train(
        # SplineKanOptions(
        #     [(28 * 28, 32), (32, 10)],
        #     2,
        #     100,
        # ),
        # HermiteKanOptions(
        #     [(28 * 28, 32), (32, 10)],
        #     5,
        # ),
        HermiteConv2dKanOptions(
            [(3, 8), (8, 16), (16, 32), (32, 64), (64, 100)],
            5,
            3,
            2,
            1,
        ),
        TrainOptions("./out/cifar100", 128, 1e-4, 1000, True),
    )


if __name__ == "__main__":
    main()
