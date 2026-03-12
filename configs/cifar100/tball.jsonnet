// bayescl/hp/cifar100/tball e0522f4
// 60.35% Acc. 2.71% ECE
// Score 78.82% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/tball.jsonnet';

dataset + method + {
  lr: 0.0025,
  peft+: {
    bnn: 'FFG',
  },
  strategy+: {
    beta: 0.62,
  },
}
