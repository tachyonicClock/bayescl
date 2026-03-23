{
  hpsearch: {
    sampler: 'BruteForceSampler',
    params: {
      'peft.rank': { type: 'categorical', choices: [1, 2, 4, 8, 16, 32, 64] },
    },
    n_trials: 10,
  },
}