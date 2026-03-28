// bayescl/hp/cifar100/clora 45cc83e 10
// Accuracy: 60.72 %
// ECE:      3.34 %
// Score:    78.69 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/clora.jsonnet';
dataset + method + {
  lr: 0.000297,
  peft+: {
    lambda_: 0.0664,
    alpha: 0.67
  }
}