// bayescl/hp/core50/ball 587aee7 166
// Accuracy: 56.09 %
// ECE:      5.87 %
// Score:    75.11 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/ball.jsonnet';
dataset + method + {
  lr: 0.000389,
  strategy+: {
    beta: 1.61,
  },
}
