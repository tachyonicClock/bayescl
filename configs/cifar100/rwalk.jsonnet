// bayescl/hp/cifar100/rwalk 9867fb2 23
// Accuracy: 64.46 %
// ECE:      11.97 %
// Score:    76.24 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/rwalk.jsonnet';
dataset + method + {
  lr: 0.00356,
  rwalk+: {
    ewc_lambda: 0.0698,
    ewc_alpha: 0.274,
  },
}
