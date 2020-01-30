# import pickle
# import time
# from audioop import reverse
# from hyperopt.fmin import generate_trials_to_calculate,generate_trial

from hyperopt import fmin, tpe, hp, STATUS_OK,Trials,trials_from_docs,space_eval
import numpy as np
def objective(args):
    print(args)
    a= args['A']
    b = args['B']
    a_val = [a[key] for key in a.keys()]
    b_val = [b[key] for key in b.keys()]

    print("a is {},b is {} and loss is {}".format(a_val,b_val,a_val[0]+b_val[0]))
    print("--------------------")
    return {'loss': a_val[0]+b_val[0], 'status': STATUS_OK }




space = {
    'A':hp.choice('A',[{'A1':hp.choice('A1',range(1,10))},{'A2':hp.uniform('A2',0,0.5)}]),
    'B':hp.choice('B',[{'B1':hp.choice('B1',range(1,10))},{'B2':hp.uniform('B2',0,0.5)}])
}

###################################
# import datetime
points = [{'A': 0, 'A1': 7, 'B': 1, 'B2': 0.45,'acc':50,'aa':20}]
print("%%%%%%%%%%%%%%%%")
# print(space_eval(space,points[0]))

##################################################################
import trial_builder_openmlspace_step3

final_trial = trial_builder_openmlspace_step3.trial_builder(points)
# final_trial=Trials()
best,trials_new = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=5,
    trials=final_trial,
    points_to_evaluate=None,
    rstate=np.random.RandomState(10),

                       )

print(len(trials_new.trials))
# print(space_eval(space,best))

