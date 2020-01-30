import openml
import pickle
import pandas as pd

#all flows on openml that have more than 1000 runs and is sklearn pipeline
flow_ids=[8817, 6969, 8815, 8890, 16345, 8317, 6970, 8315, 9666, 7707, 8351, 8353, 6952, 6840, 15083, 8774, 8786, 8918, 12736, 8844, 8834, 16360, 8330, 16347, 7096, 8795, 8797, 16374, 8887, 8365, 8399, 8885, 8793, 8788, 7116]
all_component={}



print(len(flow_ids))
valid_flows=[]
last_dic = {}
for eachflow in flow_ids:
    flow = openml.flows.get_flow(eachflow)
    flow_component = flow.components.keys()
    flowid = flow.flow_id

    if (len(flow_component)>=1) and (len(flow_component) <= 3):
        valid_flows.append(flowid)
        print(flow.class_name)
        print(flow_component)
        for key_component in flow_component:
            if key_component not in all_component:
                all_component[key_component] = [flowid]
            else:
                all_component[key_component] = all_component[key_component] + [flowid]

        print("%%%%%%%%")
        last_dic['component_step'] = list(flow_component)
        last_dic['flow_id'] = flowid

print(valid_flows)
print(len(valid_flows))
print(last_dic)
print(all_component)
print(len(all_component))