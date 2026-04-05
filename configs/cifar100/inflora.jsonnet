// bayescl/hp/cifar100/inflora 587aee7 125
// Accuracy: 54.93 %
// ECE:      3.45 %
// Score:    75.74 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/inflora.jsonnet';
dataset + method + {
  lr: 0.000387,
}
