// bayescl/hp/cifar100/linear 9867fb2
// 56.03% Acc. 2.73% ECE
// Score 76.65% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/linear.jsonnet';

dataset + method + {
  lr: 0.0016,
}
