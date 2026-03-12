// bayescl/hp/cifar100/lora 9867fb2
// 37.93% Acc. 8.44% ECE
// Score 64.74% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local base = import '../base.jsonnet';
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/lora.jsonnet';

base + dataset + method + {
  lr: 0.00011,
}
