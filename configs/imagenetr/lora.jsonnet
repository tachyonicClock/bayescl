// bayescl/hp/imagenetr/lora 9867fb2 35
// Accuracy: 29.46 %
// ECE:      7.89 %
// Score:    60.78 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/lora.jsonnet';
dataset + method + {
  lr: 0.000145,
}
