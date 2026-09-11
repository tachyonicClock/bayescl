# Experiment Design Document

Evaluate whether BALL has improved robustness in continual learning.
Robustness is quantified through resistance to forgetting, calibration, and out-of-distribution performance.

## 1. Treatments

- LoRA `lora` (control): standard deterministic low-rank adaptation.
- LoRA-Joint `lora_joint`: standard deterministic non-continual learner.
- Ensemble `lora_ensemble`: a LoRA ensemble with 5 models.
- BALL `ball` (ours)
- 3BALL `tball` (ours)
- 3BALL-MND `tball_mnd` (ours)
- C-LoRA `clora` (Smith et al., 2024)
- EWC `ewc` (Kirkpatrick et al., 2017)
- InfLoRA `inflora` (Liang & Li, 2024)
- RWalk `rwalk` (Chaudhry et al., 2018)
- SD-LoRA `sdlora` (Wu et al., 2025)

## 2. Variables

### 2.1 Response Variables

In continual learning, we care about performance after each task, not just after the final task.
Each metric is evaluated in a class- or domain-incremental manner, depending on the dataset, at the end of every task (an evaluation point): on a holdout validation set during hyperparameter tuning, and on a test set during final evaluation.
"Seen tasks" are all tasks trained on so far, including the most recent one; "future tasks" are all tasks not yet trained on.

Metrics over seen tasks are aggregated in two steps:

1. Take the mean across seen tasks at each evaluation point.
2. Take the mean across evaluation points.

When evaluated on seen tasks, early tasks contribute more since they have been seen for longer, which emphasizes remembering.
Conversely, when evaluated on future tasks, later tasks contribute more since they remain in the future for longer.
Both biases reflect the nature of the continual learning problem.

- `brier` (lower is better):
  Multi-class Brier score on seen tasks, computed as the squared error between the predicted probability vector and the one-hot label, summed over classes and averaged over samples.
  The **tuning objective**, calculated on the validation split, is used for tuning each treatment's nuisance hyperparameters.
- `acc` (higher is better): Accuracy on seen tasks.
- Calibration error (lower is better). All calibration metrics use 15 bins.
  - `nll`: Negative log-likelihood on seen tasks.
  - `ece`: Expected calibration error on seen tasks, using top-label confidence and equal-width bins.
  - `ace`: Adaptive calibration error on seen tasks, using the all-class variant with equal-mass bins (Nixon et al., 2019).
  - `ece@$shift`: Expected calibration error on seen-task samples replaced by versions with a synthetic distribution shift at a given level.
    Shift levels and types are defined in the dataset section.
  - `ace@$shift`: Like `ece@$shift`, but with `ace`.
- Out-of-distribution detection (higher is better). Calculated using the maximum softmax probability as the detection score, with in-distribution samples as the positive class.
  - `auroc_future`: AUROC for distinguishing seen-task samples from future-task samples.
    At each evaluation point, future-task samples are pooled into a single out-of-distribution set, so only the mean across evaluation points is taken.
    The final evaluation point has no future tasks, so it is excluded from the mean.
  - `auroc_$ood_dataset`: AUROC for distinguishing seen-task samples from an auxiliary out-of-distribution dataset.
    The out-of-distribution dataset is defined in the dataset section.

### 2.2 Control Variables

- **Network/LoRA architecture**. Apart from differences introduced by treatments, all architectures are the same.
- **Task order**. With the exception of `dCLEAR10/10`, whose task orderings are meaningful, task orders shall be shuffled across seeds.
- **HP search budget**. The hyperparameter search budget for each method is fixed.
  The search space itself will differ between treatments and may represent a harder or easier HPO problem.

### 2.3 Metadata

- `parameter_count`. Measure parameter counts.
  The treatments have different parameter counts. Controlling for this is not possible in this experiment. Instead, we shall measure and report the parameter count.
- `inference_time`. Measure inference time per task.
  The treatments will take different amounts of time.
- `train_time`. Measure training time per task.
  The treatments will take different amounts of time to train/converge.
- `exit_epoch`. Measure the epoch at which the model early-stops on each task.
  The treatments will converge at different rates.

### 2.4 Hyperparameter Search

Nuisance hyperparameters are controlled via hyperparameter search using Optuna (TPE search algorithm).
Each treatment configures a search space in `bayescl/treatments/$TREATMENT/_arm.py`.
The tune phase's objective is the Brier score on the holdout validation set.
During the tune phase, the task order (except for dCLEAR10/10) and initialization seeds are varied.
The validation set is also used for early stopping.

## 3. Architecture

Use a standard ResNet18 pre-trained on ImageNet in the usual way. ResNet18 is used to keep the experiments efficient to run.

To ensure models converge and to avoid overfitting while supporting different LoRA architectures, we will use early stopping on the validation Brier score, evaluated every 2 epochs with a patience of 5.

## 4. Datasets

Class-incremental and domain-incremental continual learning scenarios, constructed by splitting datasets based on classes/domains:
- `iCIFAR100/10` (10 tasks of 10 classes), based on CIFAR100.
- `iImageNet-R200/10` (10 tasks of 20 classes), based on ImageNet-R(endition).
- `dCLEAR10/10` (10 classes in 10 domains; excludes the background class).
  Note that the interpretation of `auroc_future` changes when the task is domain-incremental, since new domains are not necessarily out-of-distribution and some generalization is possible.

| Dataset      | Train  | Valid | Pilot Test | Full Test |
| ------------ | ------ | ----- | ---------- | --------- |
| `cifar100`   | 40,000 | 5,000 | 5,000      | 10,000    |
| `imagenet-r` | 25,000 | 1,250 | 1,250      | 2,500     |
| `clear10`    | 25,000 | 1,250 | 1,250      | 2,500     |

The pilot's test set shall be recycled as training data.

### 4.1 OOD Dataset Splits

| Dataset   | Pilot Test | Full Test |
| --------- | ---------- | --------- |
| `svhn`    | 5,000      | 5,000     |
| `cifar10` | 5,000      | 5,000     |

Pilot and test are a disjoint split of the original test data.

### 4.2 Dataset Shift Augmentations

[ImageNet-C-style corruptions](https://github.com/hendrycks/robustness/tree/master/ImageNet-C/imagenet_c) at 5 intensities for all datasets (Ovadia et al., 2019; Hendrycks & Dietterich, 2019).
Corruption types are sampled uniformly from the standard ImageNet-C corruption types.

| Dataset      | Pilot Test | Full Test |
| ------------ | ---------- | --------- |
| `cifar100`   | 5,000 x5   | 10,000 x5 |
| `imagenet-r` | 1,250 x5   | 2,500 x5  |
| `clear10`    | 1,250 x5   | 2,500 x5  |

Pilot and test are a disjoint split of the original test data.

## 5. Analysis

Mann-Whitney tests ($\alpha=0.05$) comparing each of our methods against each baseline and against each other, with Holm-Bonferroni multiplicity correction.
We shall compare `brier`, the mean of `auroc_$ood_dataset` over OOD datasets, and the mean of `ece@$shift` over corruption intensities.
Only these metrics were picked, to ensure statistical power under multiplicity:

```python
>>> n_our_methods = 3  # ball, tball, tball_mnd
>>> n_baselines = 6    # lora, clora, ewc, inflora, rwalk, sdlora
>>> n_datasets = 3     # cifar100, imagenet-r, clear10
>>> n_endpoints = 3    # brier, mean(auroc_$ood_dataset), mean(ece@$shift)
>>> ((n_our_methods * (n_our_methods-1))/2 + n_our_methods * n_baselines) * n_datasets * n_endpoints
189.0

```

During the pilot, the number of test runs shall be determined such that absolute differences of 0.02 in `brier`, `mean(auroc_$ood_dataset)`, and `mean(ece@$shift)` can each be detected by a two-sided Mann-Whitney test with 80% power (by convention) at the strictest Holm-corrected significance level (α = 0.05/189), using the pilot's variance estimates.
The number of runs shall be the maximum required across endpoints, datasets, and comparisons, and no fewer than 8.

## 6. Protocol

| Scale | HP Trials | Runs         | Max Epochs   |
| ----- | --------- | ------------ | ------------ |
| full  | 50        | set by pilot | set by pilot |
| pilot | 3         | 5            | 100          |

### 6.1 Pilot

- Set the appropriate number of epochs to ensure methods converge.
- Approximate standard deviations to ensure effects are likely measurable:
  - Power study.
- Validate configurations:
  - Did anything not work at all?
  - Keep costs down by running with a reduced number of HP search trials.

```bash
./main.py tune pilot $dataset $treatment
./main.py test pilot $dataset $treatment
```

### 6.2 Full

```bash
./main.py tune full $dataset $treatment
./main.py test full $dataset $treatment
```

## 7. Limitations

- **Tuning/pilot use data from all tasks:**
  In the strictest ideal of continual learning, methods would work zero-shot, without hyperparameter tuning or pilots.
  However, for research purposes, relaxing this is allowed.
- **No replay:**
  Treatments are limited to replay-free methods.
