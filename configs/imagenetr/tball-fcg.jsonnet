local dataset = import '../base/dataset/imagenetr.jsonnet';
local method = import '../base/method/tball-fcg.jsonnet';

dataset + method
