// bayescl/hp/core50/joint 9867fb2
// 44.74% Acc. 21.65% ECE
// Score 61.55% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/joint.jsonnet';

dataset + method + {
  lr: 0.0001,
}
