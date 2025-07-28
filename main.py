import matplotlib

matplotlib.use("Agg")

import argparse

from bayescl.config import from_configs
from bayescl.experiment import Experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept any number of key value pairs as dotlist arguments
    parser.add_argument("--args", type=str, nargs="*", default=None)
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=[],
        help="Path to config files to load. Can be specified multiple times.",
    )
    args = parser.parse_args()
    config = from_configs(args.configs, args.args)

    for _ in range(config.repeat):
        Experiment(config).run()  # type: ignore
