import openml
import pickle


def run_collector_specific_task(task_id,size):
    # colect the run ids for specific task
    run_ids_flow_i = []
    for key in openml.runs.list_runs(task=[task_id],size=size):
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
                last_dic['evaluations'] = run_downloaded.evaluations
                setup = openml.setups.get_setup(setup_id)
                for component in flow_component:
                    for hyperparameter in setup.parameters.values():
                        if hyperparameter.parameter_name == 'steps':
                            pass
                        else:
                            if str(component).lower() in str(hyperparameter.full_name).lower():

                                val = hyperparameter.value
                                if val in ['None']:
                                    val =None

                                elif val in ['True','False']:
                                    val =bool(val)
                                else:
                                    try:
                                        val = float(val)
                                        if val.is_integer():
                                            val = int(val)
                                    except:
                                        val =str(val)

                                last_dic['{}__{}'.format(component, hyperparameter.parameter_name)] = val
                                # last_dic['{}'.format(hyperparameter.parameter_name)] = val

    except:
        print("EXCEPT")
    return last_dic


#
# runs = run_collector_specific_task(task_id=31,size=1000)
# pickle.dump(runs, open('/home/dfki/Desktop/Thesis/openml_test/1000_runs_id_task31.p','wb'))

# runs=[2083190]
runs = [1985170, 1860342, 1860483, 1874310, 1874311, 1874312, 1874313, 1874314, 1874315, 1874316, 1874317, 1870047, 1873594, 2039491, 2036013, 2036024, 2036026, 2036029, 2036034, 2036037, 2034185, 2044888, 2016496, 2016942, 2013575, 2013577, 2015997, 2016002, 2041940, 2042009, 2081539, 2081565, 2083023, 2083110, 2083146, 2083157, 2083169, 2083171, 2083172, 2083173, 2083174, 2083175, 2083176, 2083187, 2083188, 2083189, 2083190, 2083543, 2083596]

list_runs=[]
for i in runs:
    if len(run_to_dic(i))>=1:
        list_runs.append(run_to_dic(i))
#
print(list_runs)
pickle.dump(list_runs, open('/home/dfki/Desktop/Thesis/openml_test/pickel_files/1000_list_runs_component_31.p','wb'))
