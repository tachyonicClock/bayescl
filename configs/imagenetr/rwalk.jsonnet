// bayescl/hp/imagenetr/rwalk 9867fb2 25
// Accuracy: 40.09 %
// ECE:      32.86 %
// Score:    53.62 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/rwalk.jsonnet';

dataset + method + {
  lr+: 0.000796,
  rwalk+: {
    ewc_lambda+: 0.855,
    ewc_alpha+: 0.781,
  },
}
