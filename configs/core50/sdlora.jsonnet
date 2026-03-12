local base = import '../base.jsonnet';
local dataset = import '../base/dataset/core50.jsonnet';
local method = import '../base/method/sdlora.jsonnet';

base + dataset + method
