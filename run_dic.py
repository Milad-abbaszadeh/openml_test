import openml
import pickle


def run_collector_specific_task(task_id):
    # colect the run ids for specific task
    run_ids_flow_i = []
    for key in openml.runs.list_runs(task=[task_id],size=1000):
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
            # print("Number of component is {}".format(len(flow_component)))
            # print(flow_component)

            if len(flow_component) <= 3:
                last_dic['component_step'] = list(flow_component)
                last_dic['flow_id'] = flowid
                setup = openml.setups.get_setup(setup_id)
                for component in flow_component:
                    for hyperparameter in setup.parameters.values():
                        if hyperparameter.parameter_name == 'steps':
                            pass
                        else:
                            if str(component).lower() in str(hyperparameter.full_name).lower():
                                last_dic['{}__{}'.format(component, hyperparameter.parameter_name)] = hyperparameter.value

                # print(last_dic)
                # print("$$$$$$$$$$$")
    except:
        print("EXCEPT")
    return last_dic


#
# runs = run_collector_specific_task(31)
# pickle.dump(runs, open('/home/dfki/Desktop/Thesis/openml_test/1000_runs_id_task31.p','wb'))
# list_runs=[]
# for i in runs:
#     if len(run_to_dic(i))>=1:
#         list_runs.append(run_to_dic(i))
#
# pickle.dump(list_runs, open('/home/dfki/Desktop/Thesis/openml_test/1000_list_runs_component_31.p', 'wb'))
