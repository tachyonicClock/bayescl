// bayescl/hp/imagenetr/ball 64c2033 55
// Accuracy: 51.16 %
// ECE:      4.20 %
// Score:    73.48 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/ball.jsonnet';

dataset + method + {
  lr+: 0.000536,
  strategy+: {
    beta+: 1.72,
  },
}
