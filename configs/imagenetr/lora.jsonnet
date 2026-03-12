// bayescl/hp/imagenetr/lora 9867fb2
// 29.46% Acc. 7.89% ECE
// Score 60.78% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/lora.jsonnet';

dataset + method + {
  lr: 0.00015,
}
