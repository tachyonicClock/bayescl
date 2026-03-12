{
  label+: {
    method: 'sdlora',
  },

  peft: {
    type: 'SDLoRA',
    rank_per_task: 1,  // Total rank is 10 for 10 tasks
  },
}
