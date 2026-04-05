// bayescl/hp/core50/rwalk 587aee7 153
// Accuracy: 5.95 %
// ECE:      0.69 %
// Score:    52.63 %
// Selected best run based on highest score 30 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/rwalk.jsonnet';
dataset + method + {
  lr: 0.00692,
  rwalk+: {
    ewc_alpha: 0.487,
    ewc_lambda: 0.0376,
  },
}
