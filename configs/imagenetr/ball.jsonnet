// bayescl/hp/imagenetr/ball 587aee7 156
// Accuracy: 52.42 %
// ECE:      4.18 %
// Score:    74.12 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/ball.jsonnet';
dataset + method + {
  lr: 0.000678,
  strategy+: {
    beta: 1.52,
  },
}
