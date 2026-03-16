local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/inflora.jsonnet';

dataset + method
