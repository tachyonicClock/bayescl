// bayescl/hp/core50/tball-mnd 587aee7 168
// Accuracy: 61.57 %
// ECE:      4.00 %
// Score:    78.79 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/tball-mnd.jsonnet';
dataset + method + {
  lr: 0.000549,
  strategy+: {
    beta: 0.757,
  },
}
