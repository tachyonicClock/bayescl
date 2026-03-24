local base = import 'base.jsonnet';
base {
  hpsearch+: {
    params+: {
      'strategy.beta': { type: 'float', low: 0.0, high: 2.0, step: 0.2 },
    },
  },
}
