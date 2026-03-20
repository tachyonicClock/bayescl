// bayescl/hp/cifar100/inflora 77a0dc2 47
// Accuracy: 54.39 %
// ECE:      4.05 %
// Score:    75.17 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/inflora.jsonnet';
dataset + method + {
  lr: 0.000377,
}
