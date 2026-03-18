// bayescl/hp/imagenetr/ewc 9867fb2 22
// Accuracy: 34.35 %
// ECE:      4.12 %
// Score:    65.12 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/ewc.jsonnet';

dataset + method + {
  lr+: 0.000149,
  ewc+: {
    ewc_lambda+: 9.55,
    decay_factor+: 0.883,
  },
}
