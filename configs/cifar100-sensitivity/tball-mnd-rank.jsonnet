local base = import '../cifar100/tball-mnd.jsonnet';
local search = import 'base/tball-rank.jsonnet';
base + search
