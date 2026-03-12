local base = import 'base.jsonnet';

base {
  scenario: {
    dataset: 'CIFAR100',
    n_tasks: 10,
    shuffle: false,
  },
  label: {
    scenario: 'cifar100',
  },
  epochs: 30,  // 1h budget
}
