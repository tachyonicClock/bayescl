// bayescl/hp/imagenetr/joint 9867fb2 32
// Accuracy: 63.00 %
// ECE:      1.98 %
// Score:    80.51 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/joint.jsonnet';
dataset + method + {
  lr: 0.000154,
}
