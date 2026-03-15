{
  label+: {
    method: 'si',
  },

  peft: {
    type: 'LoRA',
    r: 10,
  },

  si: {
    si_lambda: 1.0,
    eps: 0.001,
  },

  hpsearch+: {
    params+: {
      'si.si_lambda': { type: 'float', low: 0.0, high: 10.0, log: false },
      'si.eps': { type: 'float', low: 0.0001, high: 0.01, log: true },
    },
  },
}
