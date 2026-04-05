// bayescl/hp/core50/clora 587aee7 159
// Accuracy: 51.99 %
// ECE:      9.68 %
// Score:    71.16 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/clora.jsonnet';
dataset + method + {
  lr: 0.000235,
  peft+: {
    alpha: 0.694,
    lambda_: 2.9,
  },
}
