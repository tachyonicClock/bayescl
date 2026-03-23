local base = import '../cifar100/ball.jsonnet';
local search = import 'base/ball-rank.jsonnet';
base + search
