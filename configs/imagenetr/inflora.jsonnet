// bayescl/hp/imagenetr/inflora 77a0dc2 45
// Accuracy: 44.56 %
// ECE:      4.82 %
// Score:    69.87 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/inflora.jsonnet';

dataset + method + {
  lr+: 0.000341,
}
