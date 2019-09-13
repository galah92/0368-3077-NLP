from collections import deque
from trees import Node
from features import add_features_per_sample
from samples import Sample, get_state
from models import RNN
from tqdm import tqdm
import numpy as np


class Transition():

    def __init__(self, action, nuclearity=None, relation=None):
        self.action = action
        self.nuclearity = nuclearity
        self.relation = relation


def parse_files(model, trees, max_edus, vocab, infiles_dir, pred_outdir):
    pred_outdir.mkdir(exist_ok=True)
    for tree in tqdm(trees):
        queue = deque(line.strip() for line in tree.filename.with_suffix('.out.edus').open())
        stack = deque()
        root = rst_parser(queue, stack, model, vocab, tree, max_edus)
        root.to_file(pred_outdir / tree.filename.stem)


NUC_DICT = {'NN': ('Nucleus', 'Nucleus'),
            'NS': ('Nucleus', 'Satellite'),
            'SN': ('Satellite', 'Nucleus')}


def rst_parser(queue, stack, model, vocab, tree, max_edus):
    # if isinstance(model, RNN):
    #     samples = get_samples([tree])
    #     x_vecs, _, sents_idx = get_features([tree], samples, vocab)
    #     batch_size = 1
    #     input_seq = np.zeros((batch_size, model.max_seq_len, model.input_size), dtype=np.float32)
    #     input_seq[0] = model._add_padding(x_vecs, shape=(model.max_seq_len, model.input_size))
    #     actions, alter_actions = model.predict(input_seq)

    i = 0
    leaf_ind = 1
    while queue or len(stack) != 1:
        node = Node()
        node.relation = 'SPAN'

        if isinstance(model, RNN):
            transition = predict(queue, stack, model, vocab, tree, max_edus, leaf_ind, actions=(actions[i], alter_actions[i]))
        else:
            transition = predict(queue, stack, model, vocab, tree, max_edus, leaf_ind)

        if transition.action == 'SHIFT':
            node = Node(relation='SPAN',
                        text=queue.pop(),
                        span=(leaf_ind, leaf_ind))
            stack.append(node)
            leaf_ind += 1
        else:  # reduce
            right = stack.pop()
            left = stack.pop()
            node.childs.append(left)
            node.childs.append(right)
            nuclearity = NUC_DICT[transition.nuclearity]
            left.nuclearity = nuclearity[0]
            right.nuclearity = nuclearity[1]
            if left.nuclearity == 'Satellite':
                left.relation = transition.relation
            elif right.nuclearity == 'Satellite':
                right.relation = transition.relation
            else:
                left.relation = transition.relation
                right.relation = transition.relation

            if not queue and not stack:
                node.nuclearity = 'Root'
            node.span = (left.span[0], right.span[1])
            stack.append(node)
        i += 1

    return stack.pop()


def predict(queue, stack, model, vocab, tree, max_edus, top_ind_in_queue, actions=None):
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
        return Transition(action=action, nuclearity=None, relation=None)
    else:
        return Transition(*action.split('-', 2))
