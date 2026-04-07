local base = import 'base.jsonnet';
base {
  hpsearch+: {
    params+: {
      'strategy.test_samples': { type: 'categorical', choices: [1, 2, 4, 8, 16, 32] },
    },
  },
}
