// bayescl/hp/cifar100/mas 9000003 16
// Accuracy: 25.24 %
// ECE:      46.02 %
// Score:    39.61 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/mas.jsonnet';

dataset + method + {
  lr+: 0.000121,
  mas+: {
    alpha+: 0.238,
    lambda_reg+: 9.07,
  },
}
