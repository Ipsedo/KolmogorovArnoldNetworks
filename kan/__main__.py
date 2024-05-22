# -*- coding: utf-8 -*-
import argparse
import re
import sys
from typing import Dict, List, Tuple, get_args

from .options import (
    ConvOptions,
    DatasetName,
    ModelOptions,
    ResidualActivation,
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


def _parse_activations(arg: str) -> Tuple[str, Dict[str, str]]:
    args = arg.split(" ")
    assert len(args) >= 1

    act_name = args[0]
    options = {}

    for a in args[1:]:
        key, value = a.split("=", 1)
        options[key] = value

    return act_name, options


def main() -> None:
    parser = argparse.ArgumentParser("kan main")

    #########
    # Model #
    #########

    # Convolution
    parser.add_argument("channels", type=_parse_list_of_tuple)
    parser.add_argument("linear_layers", type=_parse_list_of_tuple)
    parser.add_argument(
        "-r",
        "--residual-activation",
        type=str,
        choices=get_args(ResidualActivation),
        required=True,
    )

    parser.add_argument(
        "-k", "--kernel-size", type=_parse_int_or_list_of_int, required=True
    )
    parser.add_argument(
        "-s", "--stride", type=_parse_int_or_list_of_int, required=True
    )
    parser.add_argument(
        "-p", "--padding", type=_parse_int_or_list_of_int, required=True
    )

    parser.add_argument(
        "-a",
        "--activation",
        type=str,
        required=True,
        help='Usage : "{hermite,b-spline} key_1=value_1 key_2=value_2 ..."',
    )

    ########
    # Mode #
    ########

    mode_parser = parser.add_subparsers(dest="mode", required=True)

    # train
    train_parser = mode_parser.add_parser("train")
    train_parser.add_argument(
        "dataset", type=str, choices=get_args(DatasetName)
    )
    train_parser.add_argument("dataset_path", type=str)
    train_parser.add_argument("--train-ratio", type=float, default=0.7)
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

    conv_options = ConvOptions(
        args.channels,
        args.kernel_size,
        args.stride,
        args.padding,
        args.linear_layers,
        args.residual_activation,
    )

    if args.mode == "train":
        train(
            ModelOptions(conv_options, _parse_activations(args.activation)),
            TrainOptions(
                args.dataset_path,
                args.dataset,
                args.train_ratio,
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
