// bayescl/hp/imagenetr/tball 587aee7 155
// Accuracy: 51.98 %
// ECE:      3.96 %
// Score:    74.01 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/tball.jsonnet';
dataset + method + {
  lr: 0.00113,
  strategy+: {
    beta: 0.0617,
  },
}
