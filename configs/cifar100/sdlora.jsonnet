// bayescl/hp/cifar100/sdlora 4b2f370
// 54.52% Acc. 2.93% ECE
// Score 75.80% (ACC+(1-ECE))/2
// Selected best run based on highest score 10 trials
local base = import '../base.jsonnet';
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/sdlora.jsonnet';

base + dataset + method + {
  lr: 0.0017,
}
