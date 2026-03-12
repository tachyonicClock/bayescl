// bayescl/hp/core50/tball e0522f4
// 58.76% Acc. 6.14% ECE
// Score 76.31% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/tball.jsonnet';

dataset + method + {
  lr: 0.00089,
  peft+: {
    bnn: 'FFG',
  },
  strategy+: {
    beta: 1.6,
  },
}
