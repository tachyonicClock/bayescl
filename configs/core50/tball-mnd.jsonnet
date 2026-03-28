// bayescl/hp/core50/tball-mnd db9d928 81
// Accuracy: 61.33 %
// ECE:      4.14 %
// Score:    78.60 %
// Selected best run based on highest score 10 trials
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/tball-mnd.jsonnet';
dataset + method + {
  lr: 0.000511,
  strategy+: {
    beta: 0.23
  }
}