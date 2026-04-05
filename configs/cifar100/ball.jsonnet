// bayescl/hp/cifar100/ball 587aee7 123
// Accuracy: 63.60 %
// ECE:      4.60 %
// Score:    79.50 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/ball.jsonnet';
dataset + method + {
  lr: 0.00124,
  strategy+: {
    beta: 1.82,
  },
}
