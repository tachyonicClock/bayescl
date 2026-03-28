// bayescl/hp/imagenetr/sdlora 4b2f370 12
// Accuracy: 48.23 %
// ECE:      4.71 %
// Score:    71.76 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/sdlora.jsonnet';
dataset + method + {
  lr: 0.00137
}