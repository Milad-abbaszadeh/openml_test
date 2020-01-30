from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.fmin import generate_trials_to_calculate


def run_the_strategy(args):
    print ("Current arguments:", args)
    return {
        'loss': sum([i*i for i in args]),
        'status': STATUS_OK
    }


params = [
    hp.quniform('param_1', 20, 160, 20),
    hp.quniform('param_2', 5, 100, 5),
    hp.quniform('param_3', 1, 3, 1),
    hp.quniform('param_4', 0.2, 3.0, 0.2),
    hp.choice('param_5', [0,1]),
    hp.choice('param_6', [0,1]),
    hp.choice('param_7', [0,1]),
    hp.choice('param_8', [0,1]),
    hp.choice('param_9', [0.6, 0.7, 0.8])
]

init_vals = [{'param_1': 140.0, 'param_2': 90.0, 'param_3': 1.0,
              'param_4': 2.0, 'param_5': 1, 'param_6': 1,
              'param_7': 0, 'param_8': 0, 'param_9': 0}]

trials = generate_trials_to_calculate(init_vals)

best = fmin(
            fn=run_the_strategy,
            space=params,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials
            )