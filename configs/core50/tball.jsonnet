// bayescl/hp/core50/tball e0522f4 27
// Accuracy: 58.76 %
// ECE:      6.14 %
// Score:    76.31 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/tball.jsonnet';
dataset + method + {
  lr: 0.000894,
  strategy+: {
    beta: 1.65,
  },
  peft+: {
    bnn: 'FFG',
  },
}
