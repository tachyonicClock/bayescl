// bayescl/hp/cifar100/tball-mnd 587aee7 122
// Accuracy: 63.76 %
// ECE:      3.45 %
// Score:    80.16 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/tball-mnd.jsonnet';
dataset + method + {
  lr: 0.00154,
  strategy+: {
    beta: 0.0742,
  },
}
