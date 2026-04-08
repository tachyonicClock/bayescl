local base = import '../imagenetr/ball.jsonnet';
local search = import 'base/ball-rank.jsonnet';
base + search
