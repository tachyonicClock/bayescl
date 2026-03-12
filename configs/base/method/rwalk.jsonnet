{
  label+: {
    method: 'rwalk',
  },

  peft: {
    type: 'LoRA',
    r: 10,
  },

  rwalk: {
    ewc_lambda: 0.1,
    ewc_alpha: 0.9,
    delta_t: 10,
  },

  hpsearch+: {
    params+: {
      'rwalk.ewc_lambda': { type: 'float', low: 0.0, high: 1.0, log: false },
      'rwalk.ewc_alpha': { type: 'float', low: 0.0, high: 1.0, log: false },
    },
  },
}
