// bayescl/hp/core50/inflora 77a0dc2 48
// Accuracy: 31.93 %
// ECE:      33.53 %
// Score:    49.20 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/inflora.jsonnet';

dataset + method + {
  lr+: 0.000162,
}
