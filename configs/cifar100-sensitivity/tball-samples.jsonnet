local base = import '../cifar100/tball.jsonnet';
local search = import 'base/samples.jsonnet';
base + search
