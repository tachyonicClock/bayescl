local base = import '../base.jsonnet';
local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/ewc.jsonnet';

base + dataset + method
