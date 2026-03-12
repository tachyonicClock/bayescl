// bayescl/hp/imagenetr/joint 9867fb2
// 63.00% Acc. 1.98% ECE
// Score 80.51% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/joint.jsonnet';

dataset + method + {
  lr: 0.00015,
}
