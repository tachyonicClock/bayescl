local lora = import 'lora.jsonnet';

lora {
  label+: {
    method: 'mas',
  },

  mas: {
    lambda_reg: 1.0,
    alpha: 0.5,
  },

  hpsearch+: {
    params+: {
      'mas.lambda_reg': { type: 'float', low: 0.0, high: 10.0, log: false },
      'mas.alpha': { type: 'float', low: 0.0, high: 1.0, log: false },
    },
  },
}
