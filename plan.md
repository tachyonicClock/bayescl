# Plan: Simplify ArmBase.build() to Experiment -> Experiment

Move the identical `ExperimentSpec` + `Experiment` construction boilerplate
(duplicated in all 8 arm `_arm.py` files) into a single shared factory
function. `ArmBase.build()` becomes a post-construction hook with signature
`build(self, experiment: Experiment) -> Experiment`, defaulting to
`return experiment`, instead of each arm rebuilding an `ExperimentSpec` from
scratch.

**Steps**

1. `bayescl/methods/_registry.py`: change `ArmBase.build` from the abstract
   `build(self, *, dataset, scale, seed, validation, run_dir, dataset_root) -> Experiment: raise NotImplementedError`
   to `build(self, experiment: "Experiment") -> "Experiment": return experiment`.
   Keep `base_spec()` unchanged (still used by the new factory).

2. `bayescl/experiment.py`: add a factory function (e.g. `build_experiment`)
   near the `Experiment` class:
   ```
   def build_experiment(arm, *, dataset, scale, seed, validation, run_dir, dataset_root, device="cuda") -> Experiment:
       spec = ExperimentSpec(**arm.base_spec(dataset=dataset, scale=scale, seed=seed, validation=validation, run_dir=run_dir, dataset_root=dataset_root))
       spec.device = device
       experiment = Experiment(spec, arm)
       return arm.build(experiment)
   ```
   Use `TYPE_CHECKING` imports for `ArmBase`/`Dataset`/`Scale` type hints to
   avoid circular imports (mirrors the existing pattern in `_registry.py`).
   This also folds in the `exp.spec.device = device` mutation that both
   `main.py` call sites currently do by hand.

3. *(parallel with 2, depends on 1)* Remove the duplicated `build()` override
   and now-unused `ExperimentSpec`/`Experiment` imports from each arm file:
   - `bayescl/methods/ball/_arm.py`
   - `bayescl/methods/clora/_arm.py`
   - `bayescl/methods/ewc/_arm.py`
   - `bayescl/methods/inflora/_arm.py`
   - `bayescl/methods/lora/_arm.py`
   - `bayescl/methods/rwalk/_arm.py`
   - `bayescl/methods/sdlora/_arm.py`
   - `bayescl/methods/tball/_arm.py` — also remove the dead
     `peft=self._peft()` kwarg (bug: `ExperimentSpec` has no `peft` field;
     `_build_peft` already recomputes peft config independently via
     `self._peft()`, so this was never needed).
   All 8 arms will simply inherit `ArmBase.build`'s default identity
   behaviour; none currently need to override it.

4. *(depends on 2)* `main.py`: import `build_experiment` from
   `bayescl.experiment` and update both call sites:
   - `tune()`'s `objective()` (~line 142): replace
     `arm.build(dataset=ds, scale=sc, seed=trial.number, validation=True, run_dir=..., dataset_root=Path(dataset_path))`
     + the following `exp.spec.device = device` line with a single
     `build_experiment(arm, dataset=ds, scale=sc, seed=trial.number, validation=True, run_dir=..., dataset_root=Path(dataset_path), device=device)` call.
   - `test()` loop (~line 242): same replacement pattern for the per-seed
     `arm.build(...)` + `exp.spec.device = device`.

**Relevant files**
- `bayescl/methods/_registry.py` — `ArmBase.build` signature change (abstract → default hook).
- `bayescl/experiment.py` — new `build_experiment()` factory function; reuses existing `Experiment.__init__`.
- `bayescl/methods/{ball,clora,ewc,inflora,lora,rwalk,sdlora,tball}/_arm.py` — delete `build()` overrides + unused imports; tball also drops the buggy `peft=` kwarg.
- `main.py` — both `.build(...)` call sites (tune objective, test loop) switch to `build_experiment(...)`.

**Verification**
1. `get_errors` / static check across edited files (no unused imports, no leftover `ExperimentSpec`/`Experiment` imports in arm files).
2. Run `uv run python main.py tune pilot cifar100 <method> --dataset-path <path>` for at least one non-VCL arm (e.g. `lora`) and one VCL/peft arm (`ball` or `tball`) to confirm spec construction, device assignment, and the build hook still work end-to-end.
3. Grep for `arm.build(` / `ArmBase.build` / `Experiment(spec` to confirm no stale call sites remain outside `main.py` and `experiment.py`.

**Decisions**
- Confirmed with user: `ArmBase.build(self, experiment) -> Experiment` (takes/returns an already-constructed `Experiment`), not the reverse (Experiment taking dataset/scale/etc. and calling arm internally for spec assembly).
- The `device` mutation (`exp.spec.device = device`) done ad-hoc in `main.py` is folded into the new factory's `device` parameter rather than left as a post-construction mutation.
- TBALL's `peft=self._peft()` extra `ExperimentSpec` kwarg is a pre-existing bug (no such field exists) — removed as part of this refactor rather than preserved.
- No arm currently needs to override the new `build(experiment)` hook; all 8 arms inherit `ArmBase`'s default `return experiment`. The hook exists for future arms that need post-construction customization.
