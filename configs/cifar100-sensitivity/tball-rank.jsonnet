local base = import '../cifar100/tball.jsonnet';
local search = import 'base/tball-rank.jsonnet';
base + search
