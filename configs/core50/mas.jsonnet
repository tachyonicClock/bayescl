// bayescl/hp/core50/mas 9000003 36
// Accuracy: 24.09 %
// ECE:      60.72 %
// Score:    31.68 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/mas.jsonnet';

dataset + method + {
  lr+: 0.000117,
  mas+: {
    alpha+: 0.732,
    lambda_reg+: 2.24,
  },
}
