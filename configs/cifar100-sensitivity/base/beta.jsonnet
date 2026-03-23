{
  hpsearch: {
    sampler: 'BruteForceSampler',
    params: {
      'strategy.beta': { type: 'float', low: 0.0, high: 2.0, step: 0.2 },
    },
    n_trials: 10,
  },
}
