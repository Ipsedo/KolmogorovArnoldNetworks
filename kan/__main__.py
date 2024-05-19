# -*- coding: utf-8 -*-
import argparse

from .train import SplineKanOptions, TrainOptions, train


def main() -> None:
    parser = argparse.ArgumentParser("kan main")

    _ = parser.parse_args()

    train(
        SplineKanOptions(
            [(28 * 28, 32), (32, 10)],
            3,
            100,
        ),
        TrainOptions("./out/mnist", 128, 1e-4, 100, True),
    )


if __name__ == "__main__":
    main()
