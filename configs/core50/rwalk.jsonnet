// bayescl/hp/core50/rwalk 9867fb2
// 46.36% Acc. 40.79% ECE
// Score 52.79% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/rwalk.jsonnet';

dataset + method + {
  lr: 0.002,
  rwalk+: {
    ewc_alpha: 0.76,
    ewc_lambda: 0.77,
  },
}
