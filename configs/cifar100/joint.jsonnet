// bayescl/hp/cifar100/joint 9867fb2
// 74.96% Acc. 1.00% ECE
// Score 86.98% (ACC+(1-ECE))/2
// Selected best run based on highest score 7 trials
local base = import '../base.jsonnet';
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/joint.jsonnet';

base + dataset + method + {
  lr: 0.00019,
}
