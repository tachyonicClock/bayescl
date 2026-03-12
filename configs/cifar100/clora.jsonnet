local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/clora.jsonnet';

dataset + method
