import openml
import pickle


def run_collector_specific_task(task_id,flow,size):
    # colect the run ids for specific task
    run_ids_flow_i = []
    for key in openml.runs.list_runs(task=[task_id],size=size,flow=flow):
        run_ids_flow_i.append(key)
    return run_ids_flow_i

def run_to_dic(run_id):
    last_dic = {}
    try:
        run_downloaded = openml.runs.get_run(run_id)
        setup_id = run_downloaded.setup_id
        flowid = run_downloaded.flow_id
        flow = openml.flows.get_flow(flowid)
        flow_component = flow.components.keys()
        # print(flow.class_name)

        if flow.class_name =='sklearn.pipeline.Pipeline':
            print("Number of component is {}".format(len(flow_component)))
            print(flow_component)
            print("_____")

            if len(flow_component) <= 3:
                last_dic['component_step'] = list(flow_component)
                last_dic['flow_id'] = flowid
                last_dic['evaluations'] = run_downloaded.evaluations
                setup = openml.setups.get_setup(setup_id)
                for component in flow_component:
                    for hyperparameter in setup.parameters.values():
                        if hyperparameter.parameter_name == 'steps':
                            pass
                        else:
                            if str(component).lower() in str(hyperparameter.full_name).lower():

                                val = hyperparameter.value
                                assert str(type(val)) in  ["<class 'str'>","<class 'int'>","<class 'float'>","<class 'bool'>","<class 'NoneType'>"]
                                if val in ['None','null']:
                                    val = None

                                elif val in ['True','False','true','false']:
                                    if val in ['True','true']:
                                        val = True
                                    else:
                                        val =False
                                else:
                                    try:
                                        val = float(val)
                                        if val.is_integer():
                                            val = int(val)
                                    except:

                                        if type(eval(val)) ==str:
                                            val= eval(val)
                                        else:
                                            pass
                                        # val =str(val)

                                last_dic['{}__{}'.format(component, hyperparameter.parameter_name)] = val
                                # last_dic['{}'.format(hyperparameter.parameter_name)] = val

    except:
        print("EXCEPT")
    return last_dic

flow =[8817, 6969, 8815, 8890, 16345, 8317, 6970, 8315, 9666, 7707, 8351, 8353, 6952, 6840, 15083, 8774, 8786, 8918, 12736, 8844, 8834, 16360, 8330, 16347, 7096, 8795, 16374, 8797, 8887, 8365, 8399, 8885, 8793, 8788, 7116, 16366, 7725, 17373, 8568, 7253, 7254, 8796, 17371, 13293, 7754, 7756, 7722, 6954, 7777, 8876, 7684, 7729, 8879, 7694, 8826, 8880, 7089, 6946, 7819, 16357, 17420, 7681, 8908, 7097, 8608, 8789]


runs = run_collector_specific_task(task_id=125923,flow=flow,size=None)
print(len(runs))
pickle.dump(runs, open('/home/dfki/Desktop/Thesis/openml_test/pickel_files/125923/all_sklearn_runs_id_task125923.p','wb'))

list_runs=[]
for i in runs:
    if len(run_to_dic(i))>=1:
        list_runs.append(run_to_dic(i))

print(len(list_runs))
pickle.dump(list_runs, open('/home/dfki/Desktop/Thesis/openml_test/pickel_files/125923/list_runs_component_125923_all_flow.p','wb'))
