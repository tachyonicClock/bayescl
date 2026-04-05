// bayescl/hp/core50/ewc 587aee7 152
// Accuracy: 32.74 %
// ECE:      17.41 %
// Score:    57.66 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/ewc.jsonnet';
dataset + method + {
  ewc+: {
    decay_factor: 0.9,
    ewc_lambda: 9.94,
  },
  lr: 0.000112,
}
