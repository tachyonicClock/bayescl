// bayescl/hp/imagenetr/linear 9867fb2
// 48.28% Acc. 5.05% ECE
// Score 71.61% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local base = import '../base.jsonnet';
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/linear.jsonnet';

base + dataset + method + {
  lr: 0.00063,
}
