import openml
import openmlrun_dic
import pickle

#flow ids filter base on the sklearn type and had 3component and mazimum 3 steps
flow_ids=[8817, 6969, 8815, 8890, 16345, 8317, 6970, 8315, 9666, 7707, 8351, 8353, 6952, 6840, 15083, 8774, 8786, 8918, 12736, 8844, 8834, 16360, 8330, 16347, 7096, 8795, 8797, 16374, 8887, 8365, 8399, 8885, 8793, 8788, 7116]
tasks=[31,10101,9914,145804,146065]




def flow_run(flowids,tasks,size):
    consider_runs = openml.runs.list_runs(flow=flowids,task=tasks,size=size)
    print(len(consider_runs))
    return consider_runs

def change_searchspace_to_range(searchspace):
    newspace={}
    for searchspace_key in searchspace:
        if len(searchspace[searchspace_key])>1:
           newspace[searchspace_key] = (min(searchspace[searchspace_key]),max(searchspace[searchspace_key]))
        else:
            newspace[searchspace_key] = searchspace[searchspace_key]
    print(newspace)
    return newspace


def creat_searchspace(consider_runs):
    search_space = {}
    run_dic_list = []
    for i in consider_runs:
        try:
            run_dic_list.append(openmlrun_dic.run_to_dic(i))

            if len(openmlrun_dic.run_to_dic(i))>=1:
                for key in openmlrun_dic.run_to_dic(i):
                    if key not in search_space:
                        search_space[key] = [openmlrun_dic.run_to_dic(i)[key]]
                    else:
                        if openmlrun_dic.run_to_dic(i)[key] in search_space[key]:
                            pass
                        else:
                            search_space[key] = search_space[key]+[openmlrun_dic.run_to_dic(i)[key]]
        except:
                print('except')
    print(search_space)
    return search_space


consider_runs =flow_run(flow_ids,tasks,size=None)
print(len(consider_runs))
search_space = creat_searchspace(consider_runs)
new_search_space = change_searchspace_to_range(search_space)

pickle.dump(new_search_space, open('/home/dfki/Desktop/Thesis/openml_test/searchspace.p', 'wb'))
