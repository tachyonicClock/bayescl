import argparse
import pickle
import time
from pathlib import Path
from pprint import pprint

import yaml
from avalanche.evaluation.metrics import (
    StreamConfusionMatrix,
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
    timing_metrics,
)
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from bayescl.benchmark import get_benchmark
from bayescl.config import Config


def get_logdir(cfg: Config) -> Path:
    log_dir = (
        Path(cfg.log_root) / f"{cfg.scenario}" / time.strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=False)

    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(cfg.model_dump(mode="json"), f)

    print("=" * 80)
    pprint(cfg.model_dump(mode="python"))
    print("=" * 80)
    return log_dir


def main(cfg: Config):
    # Setup logging and save config
    log_dir = get_logdir(cfg)

    benchmark = get_benchmark(cfg)

    # MODEL CREATION
    model = SimpleMLP(num_classes=benchmark.n_classes)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics, collectes their results and returns
    # them to the strategy it is attached to.

    # log to Tensorboard
    tb_logger = TensorboardLogger(log_dir)

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            experience=True,
            stream=True,
            trained_experience=True,
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        # cpu_usage_metrics(experience=True),
        forgetting_metrics(experience=True, stream=True),
        StreamConfusionMatrix(num_classes=benchmark.n_classes, save_image=True),
        # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, tb_logger],
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model=model,
        optimizer=SGD(model.parameters(), lr=0.001, momentum=0.9),
        criterion=CrossEntropyLoss(),
        train_mb_size=cfg.train_mb_size,
        eval_mb_size=cfg.eval_mb_size or cfg.train_mb_size,
        train_epochs=cfg.train_epochs,
        evaluator=eval_plugin,
        device=cfg.device,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        cl_strategy.train(experience, num_workers=cfg.num_workers)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        # eval also returns a dictionary which contains all the metric values
        results.append(
            cl_strategy.eval(benchmark.test_stream, num_workers=cfg.num_workers)
        )

    # Save results to log directory
    with open(log_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        nargs="+",
        help="yaml config file(s) to load",
    )
    args, dotlist = parser.parse_known_args()

    config = dict()  # type: ignore
    if args.config is not None:
        for file in args.config:
            config = OmegaConf.merge(config, OmegaConf.load(file))  # type: ignore
    if dotlist is not None:
        print(f"Overriding config with: {dotlist}")
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist))  # type: ignore
    main(Config(**config))  # type: ignore
