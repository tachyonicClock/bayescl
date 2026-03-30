// bayescl/hp/cifar100/tball-mnd db9d928 80
// Accuracy: 62.90 %
// ECE:      7.28 %
// Score:    77.81 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/tball-mnd.jsonnet';
dataset + method + {
  lr: 0.00381,
  strategy+: {
    beta: 0.0167,
  },
}
