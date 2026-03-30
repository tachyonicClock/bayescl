// bayescl/hp/imagenetr/tball-mnd db9d928 82
// Accuracy: 49.69 %
// ECE:      5.82 %
// Score:    71.94 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/tball-mnd.jsonnet';
dataset + method + {
  lr: 0.000648,
  strategy+: {
    beta: 0.361,
  },
}
