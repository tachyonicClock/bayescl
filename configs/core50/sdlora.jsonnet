// bayescl/hp/core50/sdlora 587aee7 161
// Accuracy: 59.49 %
// ECE:      9.47 %
// Score:    75.01 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/sdlora.jsonnet';
dataset + method + {
  lr: 0.000534,
}
