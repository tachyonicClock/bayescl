local tball = import 'tball.jsonnet';

tball {
    label+: {
        method: 'tball-fcg',
    },
    peft+: {
        bnn: 'FCG',
        rank: 5,
    }
}