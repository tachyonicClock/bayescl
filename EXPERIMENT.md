# Experiment Design Document

Evaluate if BALL has improved robustness in continual learning.
Robustness is quantified through resistance to forgetting, calibration, and out-of-distribution performance.

## 1. Treatments

- LoRA `lora` (control) standard deterministic low rank adaptation.
- BALL `ball` (ours)
- 3BALL `tball` (ours)
- 3BALL-MND `tball-mnd` (ours)
- C-LoRA `clora` (Smith et al., 2024)
- EWC `ewc` (Kirkpatrick et al., 2017)
- InfLoRA `inflora` (Liang & Li, 2024)
- RWalk (`rwalk`) (Chaudhry et al., 2018)
- SD-LoRA `sdlora` (Wu et al., 2025)

## 2. Variables
### 2.1 Response Variables

In continual learning, we care about performance after each task, not just after the final task.
Each metric is evaluated in a class-incremental manner at the end of every task (an evaluation point): on a holdout validation set during hyperparameter tuning, and on a test set during final evaluation.
"Seen tasks" are all tasks trained on so far, including the most recent one; "future tasks" are all tasks not yet trained on.

Metrics over seen tasks are aggregated in two steps:

1. take the mean across seen tasks at each evaluation point
2. take the mean across evaluation points

When evaluated on seen tasks, early tasks contribute more since they have been seen for longer, which emphasizes remembering.
Conversely, when evaluated on future tasks, later tasks contribute more since they remain in the future for longer.
Both biases reflect the nature of the continual learning problem.

#### 2.1.1 Primary Endpoint

The primary endpoint, calculated on the validation split, is used for tuning each treatment's nuisance hyperparameters.

- `brier` (lower is better): Multi-class Brier score on seen tasks, computed as the squared error between the predicted probability vector and the one-hot label, summed over classes and averaged over samples.

#### 2.1.2 Secondary Endpoints

- `acc` (higher is better): Accuracy on seen tasks.
- Calibration error (lower is better). All calibration metrics use 15 bins.
  - `nll`: Negative log-likelihood on seen-tasks.
  - `ece`: Expected calibration error on seen-tasks, using top-label confidence and equal-width bins.
  - `ace`: Adaptive calibration error on seen-tasks, using the all-class variant with equal-mass bins (Nixon et al., 2019).
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

- **Network/LoRA architecture**. All architectures apart from difference introduced by treatments are the same.
- **Task Order**. With the exception of `dCLEAR10/10` whose task orderings are meaningful, task orders shall be shuffled across seeds.

### 2.3 Confounding

- **Parameter count**

### 2.3 Nuisance Variables

## 3. Datasets

Class-incremental and a Domain-incremental continual learning scenario constructed by splitting datasets based on classes:
- `iCIFAR100/10` (10 tasks of 10 classes) based on CIFAR100.
- `iImageNet-R200`/10 (10 tasks of 20 classes) based on ImageNet-R(endition).
- `dCLEAR10/10` (10 classes in 10 domains).

| Dataset      | Train  | Valid | Pilot Test | Full Test |
| ------------ | ------ | ----- | ---------- | --------- |
| `cifar100`   | 40,000 | 5,000 | 5,000      | 10,000    |
| `imagenet-r` | 25,000 | 1,250 | 1,250      | 2,500     |
| `clear10`    | 25,000 | 1,250 | 1,250      | 2,500     |

The pilot's test set shall be recycled as training data.

### 3.1 OOD Dataset Splits

| Dataset   | Pilot Test | Full Test |
| -------   | ---------- | --------- |
| `svhn`    | 10,000     | 10,000    |
| `cifar10` | 10,000     | 10,000    |

### 3.2 Dataset shift augmentations

[`ImageNet-C` style corruptions](https://github.com/hendrycks/robustness/tree/master/ImageNet-C/imagenet_c) at 5 intensities for all datasets (Ovadia et al., 2019; Hendrycks & Dietterich, 2019).

| Dataset     | Pilot Test | Full Test |
| ----------- | ---------- | --------- |
| `cifar100`  | 5,000 x5   | 10,000 x5 |
| `imagenet-r`| 1,250 x5   | 2,500  x5 |
| `clear10`   | 1,250 x5   | 2,500  x5 |

## 4. Analysis

## 5. Protocol

- Pilot:
  - Set the appropriate number of epochs to ensure methods converge.
  - Approximate standard deviations to ensure effects are likely measurable:
    - Power Study.
  - Validate configurations.
    - Did anything not work at all?
  - Keep costs down by running with reduce hp search trials.

```bash
./main.py tune pilot $dataset $treatment
./main.py test pilot $dataset $treatment
```

- Study:

```bash
./main.py full pilot $dataset $treatment
./main.py full pilot $dataset $treatment
```
