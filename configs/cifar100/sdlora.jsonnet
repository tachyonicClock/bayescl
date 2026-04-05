// bayescl/hp/cifar100/sdlora 587aee7 126
// Accuracy: 63.36 %
// ECE:      3.39 %
// Score:    79.99 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/sdlora.jsonnet';
dataset + method + {
  lr: 0.00146,
}
