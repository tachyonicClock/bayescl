// bayescl/hp/cifar100/lora 9867fb2 33
// Accuracy: 37.93 %
// ECE:      8.44 %
// Score:    64.74 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/lora.jsonnet';
dataset + method + {
  lr: 0.000106
}