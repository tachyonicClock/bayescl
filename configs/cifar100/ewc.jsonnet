// bayescl/hp/cifar100/ewc 587aee7 118
// Accuracy: 47.02 %
// ECE:      6.86 %
// Score:    70.08 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/ewc.jsonnet';
dataset + method + {
  ewc+: {
    decay_factor: 0.863,
    ewc_lambda: 9.6,
  },
  lr: 0.000236,
}
