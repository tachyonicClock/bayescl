local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/clora.jsonnet';

dataset + method
