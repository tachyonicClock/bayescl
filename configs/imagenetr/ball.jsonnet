// bayescl/hp/imagenetr/ball 9867fb2 19
// Accuracy: 50.55 %
// ECE:      5.73 %
// Score:    72.41 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/ball.jsonnet';

dataset + method + {
  lr+: 0.000865,
  strategy+: {
    beta+: 1.06,
  },
}
