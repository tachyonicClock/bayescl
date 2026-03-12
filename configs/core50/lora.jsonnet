// bayescl/hp/core50/lora 9867fb2
// 28.37% Acc. 26.80% ECE
// Score 50.78% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local base = import '../base.jsonnet';
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/lora.jsonnet';

base + dataset + method + {
  lr: 0.00013,
}
