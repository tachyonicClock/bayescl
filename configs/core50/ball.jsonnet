// bayescl/hp/core50/ball 9867fb2
// 54.63% Acc. 8.04% ECE
// Score 73.30% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local base = import '../base.jsonnet';
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/ball.jsonnet';

base + dataset + method + {
  lr: 0.00052,
  strategy+: {
    beta: 1.5,
  },
}
