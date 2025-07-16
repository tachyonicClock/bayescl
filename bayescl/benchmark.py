from avalanche.benchmarks.classic import CORe50, SplitMNIST
from avalanche.benchmarks.scenarios import NCScenario

from bayescl.config import Config


def get_benchmark(cfg: Config) -> NCScenario:
    if cfg.scenario.dataset == "SplitMNIST":
        return SplitMNIST(
            dataset_root=cfg.dataset_root,
            n_experiences=cfg.scenario.n_tasks,
        )
    elif cfg.scenario.dataset == "CORe50":
        return CORe50(  # type: ignore
            dataset_root=cfg.dataset_root,
            run=cfg.scenario.run,
        )
    else:
        raise ValueError(f"Unsupported scenario: {cfg.scenario}")
