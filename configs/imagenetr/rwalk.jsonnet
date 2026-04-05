// bayescl/hp/imagenetr/rwalk 587aee7 169
// Accuracy: 40.07 %
// ECE:      32.21 %
// Score:    53.93 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/rwalk.jsonnet';
dataset + method + {
  lr: 0.000865,
  rwalk+: {
    ewc_alpha: 0.0869,
    ewc_lambda: 0.77,
  },
}
