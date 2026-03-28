// bayescl/hp/core50/ball 9867fb2 18
// Accuracy: 54.63 %
// ECE:      8.04 %
// Score:    73.30 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/ball.jsonnet';
dataset + method + {
  lr: 0.000522,
  strategy+: {
    beta: 1.47
  }
}