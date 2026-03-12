{
  label+: {
    method: 'clora',
  },

  peft: {
    type: 'CLoRA',
    rank: 10,
    alpha: 1.0,
    lambda_: 1.0,
  },

  hpsearch+: {
    params+: {
      'peft.lambda_': { type: 'float', low: 0.01, high: 100.0, log: true },
      'peft.alpha': { type: 'float', low: 0.5, high: 2.0 },
    },
  },
}
