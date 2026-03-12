// bayescl/hp/cifar100/rwalk 9867fb2
// 64.46% Acc. 11.97% ECE
// Score 76.24% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/rwalk.jsonnet';

dataset + method + {
  lr: 0.0036,
  rwalk+: {
    ewc_alpha: 0.27,
    ewc_lambda: 0.07,
  },
}
