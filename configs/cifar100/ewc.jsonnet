local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/ewc.jsonnet';

dataset + method
