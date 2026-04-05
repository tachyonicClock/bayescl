// bayescl/hp/imagenetr/sdlora 587aee7 160
// Accuracy: 52.19 %
// ECE:      4.23 %
// Score:    73.98 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/sdlora.jsonnet';
dataset + method + {
  lr: 0.00103,
}
