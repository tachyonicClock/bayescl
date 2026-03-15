local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/mas.jsonnet';

dataset + method
