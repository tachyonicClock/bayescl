// bayescl/hp/core50/tball 587aee7 167
// Accuracy: 62.81 %
// ECE:      4.02 %
// Score:    79.40 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/tball.jsonnet';
dataset + method + {
  lr: 0.000714,
  strategy+: {
    beta: 0.129,
  },
}
