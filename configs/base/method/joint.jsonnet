local lora = import 'lora.jsonnet';

lora {
  label+: {
    method: 'joint',
  },

  scenario+: {
    // Train only on all data at once (no continual learning)
    n_tasks: 1,
  },
}
