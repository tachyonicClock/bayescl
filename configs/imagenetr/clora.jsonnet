// bayescl/hp/imagenetr/clora 587aee7 163
// Accuracy: 53.75 %
// ECE:      4.01 %
// Score:    74.87 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/clora.jsonnet';
dataset + method + {
  lr: 0.00043,
  peft+: {
    alpha: 0.519,
    lambda_: 0.188,
  },
}
