local base = import '../cifar100/tball-mnd.jsonnet';
local search = import 'base/samples.jsonnet';
base + search
