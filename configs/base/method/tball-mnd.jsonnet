local base = import './tball.jsonnet';

base {
  label+: {
    method: 'tball-mnd',
  },
  peft+: {
    bnn: 'MND',
  },
}
