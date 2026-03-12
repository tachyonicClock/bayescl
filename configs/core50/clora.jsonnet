// bayescl/hp/core50/clora 45cc83e 11
// Accuracy: 49.37 %
// ECE:      12.92 %
// Score:    68.22 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/clora.jsonnet';

dataset + method + {
  lr+: 0.000185,
  peft+: {
    lambda_+: 8.8,
    alpha+: 0.548,
  },
}
