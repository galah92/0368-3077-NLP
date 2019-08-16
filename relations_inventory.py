RELATIONS = [
    'ATTRIBUTION',
    'BACKGROUND',
    'CAUSE',
    'COMPARISON', 
    'CONDITION',
    'CONTRAST',
    'ELABORATION',
    'ENABLEMENT',
    'TOPICCOMMENT', 
    'EVALUATION',
    'EXPLANATION',
    'JOINT',
    'MANNERMEANS',
    'SUMMARY',
    'TEMPORAL',
    'TOPICCHANGE',
    'SPAN',
    'SAME-UNIT',
    'TEXTUALORGANIZATION'
]


def build_maps():
    action_to_ind_map = {}
    ind_to_action_map = []
    ind = 0
    for relation in RELATIONS:
        for nuc in ['NN', 'NS', 'SN']:
            key = 'REDUCE-' + nuc + '-' + relation
            action_to_ind_map[key] = ind
            ind_to_action_map.append(key)
            ind += 1
    action_to_ind_map['SHIFT'] = ind
    ind_to_action_map.append('SHIFT')
    return action_to_ind_map, ind_to_action_map


action_to_ind_map, ind_to_action_map = build_maps()
