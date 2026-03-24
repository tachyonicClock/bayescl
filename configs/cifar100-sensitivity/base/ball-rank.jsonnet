local base = import 'base.jsonnet';
base {
  hpsearch+: {
    params+: {
      'peft.r': { type: 'categorical', choices: [1, 2, 4, 8, 16, 32, 64] },
    },
  },
}