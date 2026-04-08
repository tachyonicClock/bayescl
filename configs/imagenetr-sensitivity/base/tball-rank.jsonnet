local base = import 'base.jsonnet';
base {
  hpsearch+: {
    params+: {
      'peft.rank': { type: 'categorical', choices: [2, 4, 8, 16, 32, 64] },
    },
  },
}
