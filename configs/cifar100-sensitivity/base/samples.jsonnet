{
  hpsearch: {
    sampler: 'BruteForceSampler',
    params: {
      'strategy.test_samples': { type: 'categorical', choices: [1, 2, 4, 8, 16, 32, 64] },
    },
    n_trials: 10,
  },
}
