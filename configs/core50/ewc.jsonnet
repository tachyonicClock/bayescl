local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/ewc.jsonnet';

dataset + method
