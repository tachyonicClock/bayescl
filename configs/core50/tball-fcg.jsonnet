local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/tball-fcg.jsonnet';

dataset + method
