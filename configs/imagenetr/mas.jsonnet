// bayescl/hp/imagenetr/mas 9000003 37
// Accuracy: 19.43 %
// ECE:      25.06 %
// Score:    47.19 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/mas.jsonnet';

dataset + method + {
  lr+: 0.000103,
  mas+: {
    alpha+: 0.885,
    lambda_reg+: 1.95,
  },
}
