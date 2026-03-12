// bayescl/hp/imagenetr/clora 45cc83e 13
// Accuracy: 47.12 %
// ECE:      4.24 %
// Score:    71.44 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/clora.jsonnet';

dataset + method + {
  lr+: 0.000401,
  peft+: {
    lambda_+: 0.0125,
    alpha+: 0.569,
  },
}
