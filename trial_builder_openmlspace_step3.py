
from hyperopt import hp,Trials,trials_from_docs,fmin
from hyperopt.fmin import generate_trials_to_calculate,generate_trial
from hyperopt import base


def generate_trial_2(tid, space):

    acc= -space['accuracy']
    del space['accuracy']
    idxs={}
    vals={}
    for k,v in space.items():

        if space[k]=='This_is_None':
           idxs[k] =[]
           vals[k] = []
        else:
            idxs[k] =[tid]
            vals[k] = [v]


    return {
        "state": base.JOB_STATE_DONE,
        "tid": tid,
        "spec": None,
        "result": {'loss': acc, 'status': 'ok'},
        "misc": {
            "tid": tid,
            "cmd": ("domain_attachment", "FMinIter_Domain"),
            "workdir": None,
            "idxs": idxs,
            "vals": vals,
        },
        "exp_key": None,
        "owner": None,
        "version": 0,
        "book_time": None,
        "refresh_time": None,
    }


def trial_builder(points):


    new_trials = [generate_trial_2(tid, x) for tid, x in enumerate(points)]

    trials = Trials()
    final_trial = trials_from_docs(list(new_trials) + list(trials))

    return final_trial


import pickle
points_ready_turn_totrials_31 = pickle.load(open("/home/dfki/Desktop/Thesis/openml_test/pickel_files/1000_points_ready_turn_totrials_31.p", "rb"))

trial = trial_builder(points_ready_turn_totrials_31)
pickle.dump(trial, open('/home/dfki/Desktop/Thesis/openml_test/pickel_files/trial_31.p','wb'))
