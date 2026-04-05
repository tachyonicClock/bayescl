// bayescl/hp/imagenetr/tball-mnd 587aee7 154
// Accuracy: 52.72 %
// ECE:      4.47 %
// Score:    74.12 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/tball-mnd.jsonnet';
dataset + method + {
  lr: 0.00115,
  strategy+: {
    beta: 0.0262,
  },
}
