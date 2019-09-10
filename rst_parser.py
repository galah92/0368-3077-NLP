from collections import deque
from trees import Node
from features import add_features_per_sample, get_features
from samples import Sample, get_state, get_samples
from models import RNN
from tqdm import tqdm
import numpy as np


class Transition():

    def __init__(self, action, relation=None, nuclearity=None):
        self.action = action
        self.relation = relation
        self.nuclearity = nuclearity

    def gen_str(self):
        s = self.action
        if s != 'shift':
            s += '-' + ''.join(x[0] for x in self.nuclearity) + '-' + self.relation
        return s.upper()


def parse_files(model, trees, vocab, infiles_dir, pred_outdir):
    max_edus = max(tree.root.span[1] for tree in trees)
    pred_outdir.mkdir(exist_ok=True)
    for tree in tqdm(trees):
        tree_file = list(infiles_dir.glob(f'{tree.filename}*.edus'))[0]
        queue = deque(line.strip() for line in tree_file.open())
        stack = deque()
        root = rst_parser(queue, stack, model, tree, vocab, max_edus)
        root.to_file(pred_outdir / tree.filename)


def rst_parser(queue, stack, model, tree, vocab, max_edus):
    if isinstance(model, RNN):
        samples = get_samples([tree])
        x_vecs, _, sents_idx = get_features([tree], samples, vocab)
        batch_size = 1
        input_seq = np.zeros((batch_size, model.max_seq_len, model.input_size), dtype=np.float32)
        input_seq[0] = model._add_padding(x_vecs, shape=(model.max_seq_len, model.input_size))
        actions, alter_actions = model.predict(input_seq)

    i = 0
    leaf_ind = 1
    while queue or len(stack) != 1:
        node = Node()
        node.relation = 'SPAN'

        if isinstance(model, RNN):
            transition = predict(queue, stack, model, tree, vocab, max_edus, leaf_ind, actions=(actions[i], alter_actions[i]))
        else:
            transition = predict(queue, stack, model, tree, vocab, max_edus, leaf_ind)

        if transition.action == 'shift':
            node = Node(relation='SPAN',
                        text=queue.pop(),
                        span=(leaf_ind, leaf_ind))
            leaf_ind += 1
        else:
            r = stack.pop()
            l = stack.pop()
            node.childs.append(l)
            node.childs.append(r)
            l.nuclearity = transition.nuclearity[0]
            r.nuclearity = transition.nuclearity[1]
            if l.nuclearity == 'Satellite':
                l.relation = transition.relation
            elif r.nuclearity == 'Satellite':
                r.relation = transition.relation
            else:
                l.relation = transition.relation
                r.relation = transition.relation

            if not queue and not stack:
                node.nuclearity = 'Root'
            node.span = [l.span[0], r.span[1]]
        stack.append(node)
        i += 1

    return stack.pop()


NUC_DICT = {'NN': ('Nucleus', 'Nucleus'),
            'NS': ('Nucleus', 'Satellite'),
            'SN': ('Satellite', 'Nucleus')}


def predict(queue, stack, model, tree, vocab, max_edus, top_ind_in_queue, actions=None):
    state = get_state(stack, [top_ind_in_queue] if queue else [])
    sample = Sample(state=state, tree=tree)
    _, x_vecs = add_features_per_sample(sample, vocab, max_edus)
    x = np.array(x_vecs).reshape(1, -1)
    action, alter_action = actions if actions else model.predict(x)

    # correct invalid action
    if len(stack) < 2 and action != 'SHIFT':
        action = 'SHIFT'
    elif not queue and action == 'SHIFT':
        action = alter_action

    if action == 'SHIFT':
        return Transition(action='shift')
    else:
        _, nuc, relation, *_ = action.split('-')
        return Transition(action='reduce',
                          relation=relation,
                          nuclearity=NUC_DICT[nuc])
