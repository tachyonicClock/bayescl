// bayescl/hp/cifar100/ewc 9867fb2 20
// Accuracy: 43.50 %
// ECE:      5.95 %
// Score:    68.78 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/ewc.jsonnet';

dataset + method + {
  lr+: 0.000156,
  ewc+: {
    ewc_lambda+: 7.99,
    decay_factor+: 0.982,
  },
}
