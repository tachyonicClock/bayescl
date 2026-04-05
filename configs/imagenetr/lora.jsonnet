// bayescl/hp/imagenetr/lora 587aee7 164
// Accuracy: 29.47 %
// ECE:      9.35 %
// Score:    60.06 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/lora.jsonnet';
dataset + method + {
  lr: 0.000175,
}
