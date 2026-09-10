# Experiment Design Document

## Treatments

- LoRA `lora` (control) standard deterministic low rank adaptation.
- BALL `ball` (ours)
- 3BALL `tball` (ours)
- 3BALL-MND `tball-mnd` (ours)
- C-LoRA `clora` (Smith et al., 2024)
- EWC `ewc` (Kirkpatrick et al., 2017)
- InfLoRA `inflora` (Liang & Li, 2024)
- RWalk (`rwalk`) (Chaudhry et al., 2018)
- SD-LoRA `sdlora` (Wu et al., 2025)

## Datasets

## Variables
### Response Variables

In continual learning we care about performance after each task not just at the end of tasks collectively.
All response variables are micro average (each task contributes equally) of values evaluated at the end of each task.
"Seen tasks" include all previous tasks including the one that was recently trained on.
Conversely, "future tasks" are all tasks yet to be trained on.

#### Primary Endpoint

Primary endpoints calculated on the validation split are used for tuning each treatment's nuisance hyper-parameters.

- `brier`: Brier score on seen-tasks.

#### Secondary Endpoints

- `acc`: The average accuracy on seen tasks.
- `ece`: Expected calibration error on seen tasks.
- `ace`: Adaptive calibration error on seen tasks (Nixon et al., 2019).
- `auroc_future`: AUROC for future tasks. The futureless final task is not aggregated in the mean.
- `auroc_$ood_dataset`: AUROC for out-of-distribution detection using an auxiliary out-of-distribution dataset.
  The in- vs out- distribution is defined in the dataset section.
- `ece@$shift`: Expected calibration error on-seen tasks augmented with a synthetic distribution shift of a certain level.
  Shift levels and type are defined in the dataset section.
- `asce@$shift`: Like `ece@$shift` with `asce`.

### Control Variables

- **Network/LoRA architecture**. All architectures apart from difference introduced by treatments are the same.

### Confounding

- **Parameter count**

### Nuisance Variables
