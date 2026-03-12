local base = import '../base.jsonnet';
local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/ewc.jsonnet';

base + dataset + method
