// bayescl/hp/imagenetr/inflora 587aee7 158
// Accuracy: 44.79 %
// ECE:      3.66 %
// Score:    70.56 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/inflora.jsonnet';
dataset + method + {
  lr: 0.000398,
}
