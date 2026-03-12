local base = import 'base.jsonnet';

base {
  scenario: {
    dataset: 'ImageNetR',
    n_tasks: 10,
    shuffle: false,
  },
  label: {
    scenario: 'imagenetr',
  },
  epochs: 60,  // 1h budget
}
