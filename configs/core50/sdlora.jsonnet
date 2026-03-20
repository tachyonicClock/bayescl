// bayescl/hp/core50/sdlora 4b2f370 3
// Accuracy: 55.78 %
// ECE:      6.08 %
// Score:    74.85 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/sdlora.jsonnet';
dataset + method + {
  lr: 0.000275,
}
