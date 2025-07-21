import argparse

from omegaconf import OmegaConf

from bayescl.config import Config
from bayescl.experiment import Experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept any number of key value pairs as dotlist arguments
    parser.add_argument("dotlist", type=str, nargs="*")
    parser.add_argument(
        "--configs",
        "-c",
        type=str,
        nargs="*",
        default=[],
        help="Path to config files to load. Can be specified multiple times.",
    )
    args = parser.parse_args()

    config = dict()  # type: ignore
    for file in args.configs:
        config = OmegaConf.merge(config, OmegaConf.load(file))  # type: ignore
        for included in config.get("include", []):
            config = OmegaConf.merge(config, OmegaConf.load(included))
    if args.dotlist is not None:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.dotlist))  # type: ignore
    Experiment(Config.model_validate(OmegaConf.to_object(config))).run()  # type: ignore
