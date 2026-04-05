// bayescl/hp/cifar100/lora 587aee7 120
// Accuracy: 39.26 %
// ECE:      14.57 %
// Score:    62.34 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/lora.jsonnet';
dataset + method + {
  lr: 0.00021,
}
