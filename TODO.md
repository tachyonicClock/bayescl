# TODO

## `Accuracy_On_Trained_Experiences=0.0000` in early-stop eval lines is misleading

`BrierEarlyStopping.before_training_exp` calls `_check` before the new task has had
any training (`bayescl/plugin/brier_early_stopping.py:54`). `_check` evaluates
`val_stream[self._task_idx]` — the current task alone — so the model has never seen
those classes and accuracy is legitimately 0. Avalanche labels the stream summary
`Accuracy_On_Trained_Experiences`, which is wrong here: that experience is precisely
the one *not* yet trained.

The value is a benign artifact, not forgetting. It appears in every arm's log,
`lora` included, once per task boundary:

```
[early_stop] eval stream summary | Accuracy_On_Trained_Experiences=0.0000, ...
```

Worth fixing because it reads as catastrophic forgetting when skimming a run log —
easy to mistake for a real collapse, especially next to a genuinely bad run. During
the 2026-09-16 cifar100 pilots this was briefly misread as `ball` forgetting
everything at task 5.

Options, roughly in order of preference:

- Suppress the stream summary for the `eval_tag="early_stop"` context entirely; the
  line that matters is the `[early_stop] task=N epoch=M | brier=... ece=...` one
  emitted right after it.
- Relabel it under that tag to something accurate, e.g. `Accuracy_On_Current_Task`.
- Skip the `before_training_exp` `_check` call and seed `_best_brier` from the first
  post-epoch eval instead. Changes early-stopping behaviour, so only if the
  pre-training baseline turns out not to be load-bearing for the patience counter.

Emitted via `AgentLogger.after_eval` (`bayescl/metrics/agent_logger.py:176`); the
metric name comes from Avalanche, not this repo, so it has to be filtered or
rewritten on the way out.
