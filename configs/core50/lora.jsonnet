// bayescl/hp/core50/lora 587aee7 157
// Accuracy: 28.20 %
// ECE:      26.68 %
// Score:    50.76 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/lora.jsonnet';
dataset + method + {
  lr: 0.000117,
}
