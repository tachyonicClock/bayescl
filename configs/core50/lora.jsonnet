// bayescl/hp/core50/lora 9867fb2 34
// Accuracy: 28.37 %
// ECE:      26.80 %
// Score:    50.78 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/lora.jsonnet';

dataset + method + {
  lr+: 0.000132,
}
