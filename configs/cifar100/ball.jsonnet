// bayescl/hp/cifar100/ball 9867fb2 17
// Accuracy: 61.98 %
// ECE:      3.72 %
// Score:    79.13 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/ball.jsonnet';
dataset + method + {
  lr: 0.000657,
  strategy+: {
    beta: 1.92
  }
}