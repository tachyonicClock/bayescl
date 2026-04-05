// bayescl/hp/cifar100/clora 587aee7 124
// Accuracy: 61.03 %
// ECE:      3.57 %
// Score:    78.73 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/clora.jsonnet';
dataset + method + {
  lr: 0.000486,
  peft+: {
    alpha: 1.47,
    lambda_: 0.342,
  },
}
