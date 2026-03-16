{
  label+: {
    method: 'inflora',
  },

  peft: {
    type: 'InfLoRA',
    rank: 10,
    threshold_start: 0.90,
    threshold_end: 0.98,
    max_activation_batches: 16,
  },
}
