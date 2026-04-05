// bayescl/hp/cifar100/rwalk 587aee7 119
// Accuracy: 63.63 %
// ECE:      15.63 %
// Score:    74.00 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/rwalk.jsonnet';
dataset + method + {
  lr: 0.00166,
  rwalk+: {
    ewc_alpha: 0.804,
    ewc_lambda: 0.184,
  },
}
