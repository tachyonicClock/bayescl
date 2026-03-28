// bayescl/hp/cifar100/tball 0c3305e 26
// Accuracy: 60.35 %
// ECE:      2.71 %
// Score:    78.82 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/tball.jsonnet';
dataset + method + {
  lr: 0.00245,
  strategy+: {
    beta: 0.617
  },
  peft+: {
    bnn: "FFG"
  }
}