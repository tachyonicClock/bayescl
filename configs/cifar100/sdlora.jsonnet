// bayescl/hp/cifar100/sdlora 4b2f370 2
// Accuracy: 54.52 %
// ECE:      2.93 %
// Score:    75.80 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/sdlora.jsonnet';

dataset + method + {
  lr+: 0.00168,
}
