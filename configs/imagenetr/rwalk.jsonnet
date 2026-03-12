// bayescl/hp/imagenetr/rwalk 9867fb2
// 40.09% Acc. 32.86% ECE
// Score 53.62% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/rwalk.jsonnet';

dataset + method + {
  lr: 0.0008,
  rwalk+: {
    ewc_alpha: 0.78,
    ewc_lambda: 0.85,
  },
}
