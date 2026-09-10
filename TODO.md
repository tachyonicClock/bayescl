
- [X] `bayescl/search.py` implements unnecessary search space object. Replace them simply with `suggest_config` with `trial.suggests`.
- [X] Push `_build_peft`, `_build_plugins`, and `_build_strategy` into the `_arm.py` rather than the convoluted route through `ExperimentSpec`.