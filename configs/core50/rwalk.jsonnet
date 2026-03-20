// bayescl/hp/core50/rwalk 9867fb2 24
// Accuracy: 46.36 %
// ECE:      40.79 %
// Score:    52.79 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/rwalk.jsonnet';
dataset + method + {
  lr: 0.00198,
  rwalk+: {
    ewc_lambda: 0.77,
    ewc_alpha: 0.763,
  },
}
