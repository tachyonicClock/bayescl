// bayescl/hp/core50/ewc 9867fb2 21
// Accuracy: 28.92 %
// ECE:      25.26 %
// Score:    51.83 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/ewc.jsonnet';
dataset + method + {
  lr: 0.000275,
  ewc+: {
    ewc_lambda: 4.47,
    decay_factor: 0.923,
  },
}
