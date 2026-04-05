// bayescl/hp/core50/inflora 587aee7 162
// Accuracy: 33.08 %
// ECE:      32.55 %
// Score:    50.26 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/inflora.jsonnet';
dataset + method + {
  lr: 0.000101,
}
