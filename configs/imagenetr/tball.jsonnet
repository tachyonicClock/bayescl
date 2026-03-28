// bayescl/hp/imagenetr/tball 0c3305e 54
// Accuracy: 50.47 %
// ECE:      4.61 %
// Score:    72.93 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/tball.jsonnet';
dataset + method + {
  lr: 0.000966,
  strategy+: {
    beta: 0.179
  }
}