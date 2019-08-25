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

NUCLARITIES = ['NN', 'NS', 'SN']

ACTIONS = ['REDUCE-' + nuc + '-' + relation
           for relation in RELATIONS
           for nuc in NUCLARITIES] + ['SHIFT']
