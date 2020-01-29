from openmlrun_dic import run_to_dic,run_collector_specific_task
from hyperopt import fmin, tpe, hp, STATUS_OK,Trials
import openml
import copy

class Run_hyperopt(object):
    def __init__(self,dataset_id,task_id):
        self.task_id = task_id
        self.dataset_id = dataset_id

        self.task = openml.tasks.get_task(self.task_id)
        self.dataset = openml.datasets.get_dataset(dataset_id)
        self.X, self.y, self.categorical_indicator, attribute_names = self.dataset.get_data(
            dataset_format='array',
            target=self.dataset.default_target_attribute
        )
        self.copy_X = copy.deepcopy(self.X)
        self.copy_y = copy.deepcopy(self.y)
        self.time_tracker=[]

    def rest_x_y(self):
        self.X = self.copy_X
        self.y = self.copy_y

    def make_search_space(self):
        search_space={
            'data_preprocessing':hp.choice('data_preprocessing',[
                {'type':'Normalizer'},
                {'type':'SimpleImputer'},
                {'type':'ColumnTransformer',
                 'remainder':hp.choice('ColumnTransformer__remainder',["drop", "passthrough"])},
                {'type':'standard_scaler'},
                {'type':'minmaxscaler'},
                {'type':"do_noting"}
            ]),

            'feature_preprocessing':hp.choice('feature_preprocessing',[
                    {'type':'pca',
                    'pca__iterated_power': hp.choice('pca__iterated_power', ['auto']+list(range(1,10))),
                    'pca__n_components': hp.choice('pca__n_components',[None] +list(range(1,self.X.shape[1]))),
                    'pca__svd_solver':hp.choice('pca__svd_solver',['auto','full','randomized']),
                    'pca__tol': hp.uniform('pca__tol',0,0.5),
                    'pca__whiten': hp.choice('pca__whiten', [True, False])},

                    {'type':'kernel_pca',
                     # 'kernel':hp.choice('kernel',["linear","poly","rbf","sigmoid","cosine"]),
                     'kernel_pca__n_components':hp.choice('kernel_pca__n_components',range(10,self.X.shape[1]))
                     },

                    {   'type':'VarianceThreshold',
                        'VarianceThreshold__threshold':hp.uniform('VarianceThreshold__threshold',0,0.5)
                    },

                    {'type':"do_noting"}

            ]),

            'classifier':hp.choice('classifier',[
                {'type':'randomforestclassifier',
                'randomforestclassifier__criterion': hp.choice('randomforestclassifier__criterion', ["gini", "entropy"]),
                'randomforestclassifier__max_depth': hp.choice('randomforestclassifier__max_depth', [None] +list(range(2,1000))),
                'randomforestclassifier__min_samples_leaf': hp.choice('randomforestclassifier__min_samples_leaf', range(1,20)),
                'randomforestclassifier__min_samples_split': hp.choice('randomforestclassifier__min_samples_split', range(2,20)),
                'randomforestclassifier__min_weight_fraction_leaf': hp.uniform('randomforestclassifier__min_weight_fraction_leaf', 0.0, 0.5),
                'randomforestclassifier__max_features': hp.choice('randomforestclassifier__max_features', ['auto', 'sqrt', 'log2',0.15,0.25,0.3,0.4,0.45]),
                'randomforestclassifier__n_estimators': hp.choice('randomforestclassifier__n_estimators', range(10,1000)),
                'randomforestclassifier__oob_score': hp.choice('randomforestclassifier__oob_score', [True, False]),
                 },

                {'type':'decisiontreeclassifier',
                 'decisiontreeclassifier__criterion': hp.choice('decisiontreeclassifier__criterion', ["gini", "entropy"]),
                 'decisiontreeclassifier__max_depth':hp.choice('decisiontreeclassifier__max_depth',[0.0009035528566034845,None]),
                 'decisiontreeclassifier__min_samples_leaf': hp.choice('decisiontreeclassifier__min_samples_leaf', range(1, 9)),
                 'decisiontreeclassifier__min_samples_split': hp.choice('decisiontreeclassifier__min_samples_split', [9,10]),
                 },

                {'type':'gradientboostingclassifier',
                 'gradientboostingclassifier__criterion': hp.choice('gradientboostingclassifier__criterion', ["friedman_mse", "mse"]),
                 'gradientboostingclassifier__learning_rate':hp.uniform('gradientboostingclassifier__learning_rate',9.920058705184867e-05,0.00010056450840281946),
                 'gradientboostingclassifier__max_depth': hp.choice('gradientboostingclassifier__max_depth', range(1,9)),
                 'gradientboostingclassifier__max_features':hp.uniform('gradientboostingclassifier__max_features', 0.00015525642662705952, 0.9998642646284683),
                 'gradientboostingclassifier__min_impurity_decrease':hp.uniform('gradientboostingclassifier__min_impurity_decrease',0.00022898940251292466, 0.9996576747926129),
                 'gradientboostingclassifier__min_samples_leaf':hp.choice('gradientboostingclassifier__min_samples_leaf',range(1,9)),
                 'gradientboostingclassifier__min_samples_split':hp.choice('gradientboostingclassifier__min_samples_split',[2,9,10]),
                 'gradientboostingclassifier__min_weight_fraction_leaf':hp.uniform('gradientboostingclassifier__min_weight_fraction_leaf',8.873194131375772e-05,0.0001884133057376003),
                'gradientboostingclassifier__n_estimators':hp.choice('gradientboostingclassifier__n_estimators',range(100,995)),
                 'gradientboostingclassifier__n_iter_no_change':hp.choice('gradientboostingclassifier__n_iter_no_change',range(1,999)),
                 'gradientboostingclassifier__subsample':hp.uniform('gradientboostingclassifier__subsample',9.236456951389194e-06,0.0002081432615039791),
                 'gradientboostingclassifier__tol':hp.uniform('gradientboostingclassifier__tol',9.996741607059855e-05,0.0001001692053800057),
                 'gradientboostingclassifier__validation_fraction':hp.uniform('gradientboostingclassifier__validation_fraction',0.00027270272088730785, 0.99676753787075),
                },

                {'type':'bernoullinb',
                 'bernoullinb__fit_prior':hp.choice('bernoullinb__fit_prior',[True,False]),
                 'bernoullinb__alpha':hp.uniform('bernoullinb__alpha', 0.010073368015954882, 98.93346969207758),

                },
                {'type':'fkceigenpro',
                'fkceigenpro__degree':hp.choice('fkceigenpro__degree',range(2,4)),
                 'fkceigenpro__gamma':hp.uniform('fkceigenpro__gamma', 1e-10,0.0001),
                 'fkceigenpro__kernel':hp.choice('fkceigenpro__kernel',["laplace", "rbf"]),
                 'fkceigenpro__n_components':hp.choice("fkceigenpro__n_components",range(500,1000))

                },

                {'type':'svc',
                 'svc__C':hp.uniform("svc__C", 0.01,9979.44679282882),
                 'svc__coef0':hp.uniform('svc__coef0',-0.0001901088806708362, 0.9996939328918386),
                 'svc__degree':hp.choice('svc__degree',range(1,5)),
                 'svc__gamma':hp.uniform('svc__gamma',9.984514749387293e-05,0.00010001864000043732),
                 'svc__kernel':hp.choice('svc__kernel',["linear", "sigmoid",'rbf']),
                 'svc__shrinking':hp.choice('svc__shrinking',[True,False]),
                 'svc__tol':hp.uniform('svc__tol',9.990234352037583e-05, 0.00010032523263523512),
                },

                {'type':'kneighborsclassifier',
                 'kneighborsClassifier__n_neighbors':hp.choice('kneighborsClassifier__n_neighbors',range(2,10)),
                 'kneighborsClassifier__algorithm':hp.choice('kneighborsClassifier__algorithm',['auto', 'ball_tree', 'kd_tree', 'brute'])
                 },

                {'type':'extratreesclassifier',
                 'extratreesclassifier__bootstrap':hp.choice('extratreesclassifier__bootstrap',[True,False]),
                 'extratreesclassifier__criterion': hp.choice('extratreesclassifier__criterion', ["gini", "entropy"]),
                 'extratreesclassifier__max_features': hp.uniform('extratreesclassifier__max_features',
                                            0.00296553169445235, 0.9884684507203433),
                 'extratreesclassifier__min_samples_leaf': hp.choice('extratreesclassifier__min_samples_leaf', range(1, 9)),
                 'extratreesclassifier__min_samples_split': hp.choice('extratreesclassifier__min_samples_split', [9, 10]),
                },
                {'type':'mlpclassifier',
                 'mlpclassifier__activation':hp.choice('mlpclassifier__activation',["identity", "tanh"]),
                 'mlpclassifier__alpha':hp.uniform('mlpclassifier__alpha', 6.541497552990362e-05,0.00010695575243507994),
                 'mlpclassifier__batch_size':hp.choice('mlpclassifier__batch_size',['auto',95]),
                 'mlpclassifier__beta_1':hp.uniform('mlpclassifier__beta_1', 0.00544424784765507, 0.9),
                 'mlpclassifier__beta_2': hp.uniform('mlpclassifier__beta_2', 0.047423225221743026, 0.999),
                'mlpclassifier__early_stopping':hp.choice('mlpclassifier__early_stopping',[True,False]),
                 'mlpclassifier__hidden_layer_sizes':hp.choice('mlpclassifier__hidden_layer_sizes',range(951,1013)),
                 'mlpclassifier__learning_rate':hp.choice('mlpclassifier__learning_rate',["adaptive", "invscaling"]),
                 'mlpclassifier__learning_rate_init':hp.uniform('mlpclassifier__learning_rate_init',7.740530907783659e-05,0.00013450694347599834),
                 'mlpclassifier__max_iter':hp.choice('mlpclassifier__max_iter',range(992,1003)),
                 'mlpclassifier__momentum':hp.uniform('mlpclassifier__momentum',0.06610188576749942, 0.983051121954481),
                 'mlpclassifier__n_iter_no_change':hp.choice('mlpclassifier__n_iter_no_change',range(10,987)),
                 'mlpclassifier__nesterovs_momentum':hp.choice('mlpclassifier__nesterovs_momentum',[True,False]),
                 'mlpclassifier__power_t':hp.uniform('mlpclassifier__power_t',5.7659652445073064e-05,0.0002094262206310496),
                 'mlpclassifier__shuffle':hp.choice('mlpclassifier__shuffle',[True,False]),
                 'mlpclassifier__solver':hp.choice('mlpclassifier__solver',["adam", "sgd"]),
                 'mlpclassifier__tol':hp.uniform('mlpclassifier__tol',7.072577204620778e-05,0.0001),
                },

                {'type':'sgdclassifier',
                 'sgdclassifier__loss':hp.choice('sgdclassifier__loss',["log", "modified_huber", "squared_hinge", "perceptron"]),
                 'sgdclassifier__penalty':hp.choice('sgdclassifier__penalty',["l1", "l2", "elasticnet"]),
                 'sgdclassifier__alpha':hp.uniform('sgdclassifier__alpha',1e-7,1e-1),
                 'sgdclassifier__max_iter':hp.choice('sgdclassifier__max_iter',range(5,1000)),
                 'sgdclassifier__tol':hp.uniform('sgdclassifier__tol',1e-5, 1e-1)}
            ])

        }
        return search_space

def give_searchspace(dataset_id,task_id):
    runner = Run_hyperopt(dataset_id, task_id)
    search_space = runner.make_search_space()
    return search_space


# runs_task_i = run_collector_specific_task(task_id=31,size=10)
#fifty runs for task i
# runs_task_i = [1985170, 1860342, 1860483, 1874310, 1874311, 1874312, 1874313, 1874314, 1874315, 1874316, 1874317, 1861204, 1870047, 1873594, 2039491, 2036013, 2036024, 2036026, 2036029, 2036034, 2036037, 2034185, 2044888, 2016496, 2016942, 2013575, 2013577, 2015997, 2016002, 2041940, 2042009, 2081539, 2081565, 2083023, 2083110, 2083146, 2083157, 2083169, 2083171, 2083172, 2083173, 2083174, 2083175, 2083176, 2083187, 2083188, 2083189, 2083190, 2083543, 2083596]
#
# runner = Run_hyperopt(31,31)
#
# search_space = runner.make_search_space()

# new_list_runs = []
# for index in range(len(runs_task_i)):
#     new_list_runs.append(change_dic_hyperoptobj(search_space, run_to_dic(runs_task_i[index])))
#
# print(new_list_runs)