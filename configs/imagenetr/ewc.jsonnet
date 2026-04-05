// bayescl/hp/imagenetr/ewc 587aee7 165
// Accuracy: 35.53 %
// ECE:      4.10 %
// Score:    65.71 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/ewc.jsonnet';
dataset + method + {
  ewc+: {
    decay_factor: 0.963,
    ewc_lambda: 9.26,
  },
  lr: 0.00017,
}
