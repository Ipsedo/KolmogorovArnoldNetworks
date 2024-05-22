# -*- coding: utf-8 -*-
import argparse
import re
import sys
from typing import List, Tuple, get_args

from .options import (
    ActivationsOptions,
    ArchitectureOptions,
    ConvOptions,
    HermiteOptions,
    LinearOptions,
    ModelOptions,
    ResidualActivation,
    SplineOptions,
    TrainOptions,
)
from .train import train


def _parse_list_of_tuple(string: str) -> List[Tuple[int, int]]:
    regex_match = re.compile(
        r"^ *\[(?: *\( *\d+ *, *\d+ *\) *,)* *\( *\d+ *, *\d+ *\) *] *$"
    )
    regex_layer = re.compile(r"\( *\d+ *, *\d+ *\)")
    regex_space = re.compile(r"\d+")

    assert regex_match.match(string), "usage : [(10, 20), (20, 40), ...]"

    def _match_spaces(layer_str: str) -> Tuple[int, int]:
        matched = regex_space.findall(layer_str)
        assert len(matched) == 2
        return int(matched[0]), int(matched[1])

    return [_match_spaces(layer) for layer in regex_layer.findall(string)]


def _parse_list_of_int(string: str) -> List[int]:
    regex_true_false = re.compile(r"\d+")
    regex_match = re.compile(r"^ *\[(?: *\d+ *,)* *\d+ *] *$")

    assert regex_match.match(string), "usage : [2, 3, 1, ...]"

    return [int(use_att) for use_att in regex_true_false.findall(string)]


def _parse_int_or_list_of_int(string: str) -> List[int] | int:
    regex_integer = re.compile(r"^ *\d+ *$")

    if regex_integer.match(string):
        return int(string)

    return _parse_list_of_int(string)


def main() -> None:
    parser = argparse.ArgumentParser("kan main")

    #########
    # Model #
    #########

    arch_parser = parser.add_subparsers(dest="architecture", required=True)

    # Linear
    linear_parser = arch_parser.add_parser("linear")
    linear_parser.add_argument("layers", type=_parse_list_of_tuple)
    linear_parser.add_argument(
        "-r",
        "--residual-activation",
        type=str,
        choices=get_args(ResidualActivation),
        required=True,
    )

    # Convolution
    conv_parser = arch_parser.add_parser("conv")
    conv_parser.add_argument(
        "channels", type=_parse_list_of_tuple, required=True
    )
    conv_parser.add_argument(
        "linear-layers", type=_parse_list_of_tuple, required=True
    )
    conv_parser.add_argument(
        "-r",
        "--residual-activation",
        type=str,
        choices=get_args(ResidualActivation),
        required=True,
    )

    conv_parser.add_argument(
        "-k", "--kernel-size", type=_parse_int_or_list_of_int, required=True
    )
    conv_parser.add_argument(
        "-s", "--stride", type=_parse_int_or_list_of_int, required=True
    )
    conv_parser.add_argument(
        "-p", "--padding", type=_parse_int_or_list_of_int, required=True
    )

    ##############
    # Activation #
    ##############

    act_parser = parser.add_subparsers(dest="activation", required=True)

    # spline
    spline_parser = act_parser.add_parser("spline")
    spline_parser.add_argument("-d", "--degree", type=int, default=2)
    spline_parser.add_argument("-g", "--grid-size", type=int, default=8)

    # hermite
    hermite_parser = act_parser.add_parser("hermite")
    hermite_parser.add_argument("-n", "--n-hermite")

    ########
    # Mode #
    ########

    mode_parser = parser.add_subparsers(dest="mode", required=True)

    # train
    train_parser = mode_parser.add_parser("train")
    train_parser.add_argument("dataset_path", type=str)
    train_parser.add_argument("-e", "--epochs", type=int, default=100)
    train_parser.add_argument("-b", "--batch-size", type=int, default=24)
    train_parser.add_argument(
        "-lr", "--learning-rate", type=float, default=1e-4
    )
    train_parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    ################
    # Main Program #
    ################

    act_options: ActivationsOptions
    if args.activation == "spline":
        act_options = SplineOptions(args.degree, args.grid_size)
    elif args.activation == "hermite":
        act_options = HermiteOptions(args.n_hermite)
    else:
        parser.error(f"Unknown activation {args.activation}")
        sys.exit(1)

    arch_options: ArchitectureOptions
    if args.architecture == "linear":
        arch_options = LinearOptions(args.layers, args.residual_activation)
    elif args.architecture == "conv":
        arch_options = ConvOptions(
            args.channels,
            args.kernel_size,
            args.stride,
            args.padding,
            args.linear_layers,
            args.residual_activation,
        )
    else:
        parser.error(f"Unknown architecture {args.architecture}")
        sys.exit(1)

    if args.mode == "train":
        train(
            ModelOptions(arch_options, act_options),
            TrainOptions(
                args.dataset_path,
                args.batch_size,
                args.learning_rate,
                args.epochs,
                args.cuda,
            ),
        )
    else:
        parser.error(f"Unknown mode {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
