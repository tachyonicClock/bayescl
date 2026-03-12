// bayescl/hp/imagenetr/ball 9867fb2
// 50.55% Acc. 5.73% ECE
// Score 72.41% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local base = import '../base.jsonnet';
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/ball.jsonnet';

base + dataset + method + {
  lr: 0.00087,
  strategy+: {
    beta: 1.1,
  },
}
