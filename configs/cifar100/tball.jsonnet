// bayescl/hp/cifar100/tball 587aee7 121
// Accuracy: 63.75 %
// ECE:      3.12 %
// Score:    80.31 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/tball.jsonnet';
dataset + method + {
  lr: 0.00223,
  strategy+: {
    beta: 0.00486,
  },
}
