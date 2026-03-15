local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/si.jsonnet';

dataset + method
