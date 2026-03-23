local base = import '../cifar100/ball.jsonnet';
local search = import 'base/samples.jsonnet';
base + search
