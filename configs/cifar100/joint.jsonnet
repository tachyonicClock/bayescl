// bayescl/hp/cifar100/joint 9867fb2 29
// Accuracy: 74.96 %
// ECE:      1.00 %
// Score:    86.98 %
// Selected best run based on highest score 7 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/joint.jsonnet';
dataset + method + {
  lr: 0.000195,
}
