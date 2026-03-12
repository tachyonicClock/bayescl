// bayescl/hp/cifar100/ball 9867fb2
// 61.98% Acc. 3.72% ECE
// Score 79.13% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/ball.jsonnet';

dataset + method + {
  lr: 0.00066,
  strategy+: {
    beta: 1.9,
  },
}
