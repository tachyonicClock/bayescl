{
  hpsearch: {
    sampler: 'BruteForceSampler',
    n_trials: 100,  // Will stop earlier if all combinations are exhausted
    direction: ['maximize', 'minimize'],
    params: {
      seed: { type: 'categorical', choices: [0, 1, 2, 3, 4] },
    },
  },
}
