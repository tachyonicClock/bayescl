{
  label+: {
    method: 'tball',
  },

  // BALL uses local CE but it is implemented internally so we turn this off here
  use_local_ce: false,

  peft: {
    type: 'TBALL',
    rank: 10,
    init_sd: 0.5,
    prior_weight_sd: 1.0,
    bias: false,
    bnn: 'FFG',
  },

  strategy: {
    type: 'VCL',
    beta: 1.0,
    train_samples: 1,
    test_samples: 5,
  },

  hpsearch+: {
    params+: {
      'strategy.beta': { type: 'float', low: 0.0, high: 2.0, log: false },
    },
  },
}
