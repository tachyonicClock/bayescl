// bayescl/hp/core50/joint 9867fb2 31
// Accuracy: 44.74 %
// ECE:      21.65 %
// Score:    61.55 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/joint.jsonnet';
dataset + method + {
  lr: 0.0001,
}
