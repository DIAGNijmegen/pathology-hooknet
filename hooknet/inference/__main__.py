import argparse
import signal
import sys
from pathlib import Path
import time
from hooknet.inference.apply import execute_inference


def signal_handler(*args):
    print("Exit gracefully...")
    sys.exit(0)


def _parse_args():
    # create argument parser
    argument_parser = argparse.ArgumentParser(description="Experiment")
    argument_parser.add_argument("-u", "--user_config", required=True)
    argument_parser.add_argument("-n", "--model_name", required=True)
    argument_parser.add_argument("-o", "--output_folder", required=True)
    argument_parser.add_argument("-d", "--tmp_folder", required=True)
    argument_parser.add_argument("-m", "--mode", required=False)
    argument_parser.add_argument("-s", "--source_preset", required=False)
    argument_parser.add_argument("-c", "--cpus", required=False)
    argument_parser.add_argument(
        "-t", "--heatmaps", type=int, nargs="+", required=False
    )
    args = vars(argument_parser.parse_args())

    args["output_folder"] = Path(args["output_folder"])
    args["tmp_folder"] = Path(args["tmp_folder"])

    if "mode" not in args or not args["mode"]:
        args["mode"] = "default"

    if "source_preset" not in args or not args["source_preset"]:
        args["source_preset"] = "folders"
    else:
        args["source_preset"] = args["source_preset"]

    if "cpus" not in args or not args["cpus"]:
        args["cpus"] = 1
    else:
        args["cpus"] = int(args["cpus"])

    if "heatmaps" not in args or not args["heatmaps"]:
        args["heatmaps"] = None

    return args


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    args = _parse_args()
    execute_inference(
        user_config=args["user_config"],
        mode=args["user_config"],
        model_name=args["model_name"],
        output_folder=args["output_folder"],
        tmp_folder=args["tmp_folder"],
        cpus=args["cpus"],
        source_preset=args["source_preset"],
        heatmaps=args["heatmaps"],
    )


if __name__ == "__main__":
    main()
