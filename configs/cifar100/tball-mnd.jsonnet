local dataset = import '../base/dataset/cifar100.jsonnet';
local method = import '../base/method/tball-mnd.jsonnet';
dataset + method
