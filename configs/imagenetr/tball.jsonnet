// bayescl/hp/imagenetr/tball e0522f4
// 46.58% Acc. 10.43% ECE
// Score 68.07% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/tball.jsonnet';

dataset + method + {
  lr: 0.0036,
  strategy+: {
    beta: 0.41,
  },
}
