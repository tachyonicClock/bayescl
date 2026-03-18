// bayescl/hp/imagenetr/tball e0522f4 28
// Accuracy: 46.58 %
// ECE:      10.43 %
// Score:    68.07 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/tball.jsonnet';

dataset + method + {
  lr+: 0.00358,
  strategy+: {
    beta+: 0.408,
  },
}
