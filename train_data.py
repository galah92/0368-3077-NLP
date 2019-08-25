import copy


class Sample():

    def __init__(self):
        self.state = []  # [v1, v2, v3] where v1 & v2 are the elements at the top of the stack
        self.action = ''
        self.tree = ''


def gen_train_data(trees):
    samples = []
    for tree in trees:
        root = tree._root
        stack = []
        tree_samples = []
        queue = list(range(tree._root.span[1], 0, -1))  # queue of EDUS indices
        gen_train_data_tree(root, stack, queue, tree_samples)
        tree._samples = copy.copy(tree_samples)
        for sample in tree_samples:
            sample.tree = tree
            samples.append(sample)
    return samples


def gen_train_data_tree(node, stack, queue, samples):
    sample = Sample()
    if not node.childs:
        sample.action = 'SHIFT'
        sample.state = genstate(stack, queue)
        assert(queue.pop(-1) == node.span[0])
        stack.append(node)
    else:
        l, r = node.childs
        gen_train_data_tree(l, stack, queue, samples)
        gen_train_data_tree(r, stack, queue, samples)
        if r.nuclearity == 'Satellite':
            sample.action = genaction(node, r)
        else:
            sample.action = genaction(node, l)
        sample.state = genstate(stack, queue)
        assert(stack.pop(-1) == node.childs[1])
        assert(stack.pop(-1) == node.childs[0])
        stack.append(node)
    samples.append(sample)


def genaction(parent, child):
    if child.nuclearity == 'Satellite':
        nuc = 'SN' if parent.childs[0] == child else 'NS'
    else:
        nuc = 'NN'
    return f'REDUCE-{nuc}-{child.relation}'


def genstate(stack, queue):
    ind1, ind2, ind3 = 0, 0, 0
    if len(queue) > 0:
        ind3 = queue[-1]
    if len(stack) > 0:
        ind1 = get_nuclear_edu_ind(stack[-1])  # right son
        if len(stack) > 1:
            ind2 = get_nuclear_edu_ind(stack[-2])  # left son
    return ind1, ind2, ind3


def get_nuclear_edu_ind(node):
    if not node.childs:
        return node.span[0]
    left = node.childs[0]
    right = node.childs[1]
    if left.nuclearity == 'Nucleus':
        return get_nuclear_edu_ind(left)
    return get_nuclear_edu_ind(right)
