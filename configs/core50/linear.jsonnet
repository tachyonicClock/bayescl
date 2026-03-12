// bayescl/hp/core50/linear 9867fb2
// 53.14% Acc. 4.55% ECE
// Score 74.30% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/linear.jsonnet';

dataset + method + {
  lr: 0.00044,
}
