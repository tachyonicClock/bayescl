local base = import 'base.jsonnet';

base {
  scenario: {
    dataset: 'CORe50',
    n_tasks: 10,
    shuffle: false,
  },
  label: {
    scenario: 'core50',
  },
  epochs: 30,  // 1h budget
}
