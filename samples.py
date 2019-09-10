from collections import deque
import copy


class Sample():

    def __init__(self, state):
        self.state = state
        self.action = ''
        self.tree = ''


def get_samples(trees):
    samples = []
    for tree in trees:
        root = tree.root
        stack = deque()
        tree_samples = []
        queue = deque(range(tree.root.span[1], 0, -1))  # queue of EDUS indices
        get_samples_rec(root, stack, queue, tree_samples)
        tree._samples = copy.copy(tree_samples)
        for sample in tree_samples:
            sample.tree = tree
            samples.append(sample)
    return samples


def get_samples_rec(node, stack, queue, samples):
    sample = Sample(get_state(stack, queue))
    if not node.childs:
        sample.action = 'SHIFT'
        queue.pop()
    else:
        left, right = node.childs
        child = right if right.nuclearity == 'Satellite' else left
        sample.action = get_action(node, child)
        get_samples_rec(left, stack, queue, samples)
        get_samples_rec(right, stack, queue, samples)
        stack.pop()
        stack.pop()
    stack.append(node)
    samples.append(sample)


def get_action(parent, child):
    if child.nuclearity == 'Satellite':
        nuc = 'SN' if parent.childs[0] == child else 'NS'
    else:
        nuc = 'NN'
    return f'REDUCE-{nuc}-{child.relation}'


def get_state(stack, queue):
    ind1, ind2, ind3 = 0, 0, 0
    if len(queue) > 0:
        ind3 = queue[-1]
    if len(stack) > 0:
        ind1 = stack[-1].get_edu_ind()  # right son
        if len(stack) > 1:
            ind2 = stack[-2].get_edu_ind()  # left son
    return ind1, ind2, ind3
