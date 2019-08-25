from collections import deque
from preprocess import Node
from evaluation import eval as evaluate
from features import add_features_per_sample
from train_data import Sample, genstate
from relations import ACTIONS
from tqdm import tqdm
import numpy as np
import torch


class Transition():

    def __init__(self):
        self.nuclearity = []  # <nuc>, <nuc>
        self.relation = ''  # cluster relation
        self.action = ''  # shift or 'reduce'

    def gen_str(self):
        s = self.action
        if s != 'shift':
            s += '-' + ''.join(x[0] for x in self.nuclearity) + '-' + self.relation
        return s.upper()


def parse_files(model_name, model, trees, vocab, infiles_dir, gold_files_dir, pred_outdir):
    max_edus = max(tree._root.span[1] for tree in trees)
    pred_outdir.mkdir(exist_ok=True)
    for tree in tqdm(trees):
        tree_file = list(infiles_dir.glob(f'{tree._fname}*.edus'))[0]
        queue = deque(line.strip() for line in tree_file.open())
        stack = deque()
        root = parse_file(queue, stack, model_name, model, tree, vocab, max_edus)
        root.to_file(pred_outdir / tree._fname)
    evaluate(gold_files_dir, pred_outdir)


def parse_file(queue, stack, model_name, model, tree, vocab, max_edus):
    ## RNN ##
    # samples, _ = gen_train_data([tree])
    # x_vecs, _, sents_idx = get_features([tree], samples, vocab)
    # batch_size = 1
    # input_seq = np.zeros((batch_size, model.max_seq_len, model.input_size), dtype=np.float32)
    # input_seq[0] = add_padding(x_vecs, shape=(model.max_seq_len, model.input_size))
    # actions, _ = rnn_predict(model, input_seq)
    # actions = actions#[:len(x_vecs)+1]
    ######

    leaf_ind = 1
    while queue or len(stack) != 1:
        node = Node()
        node.relation = 'SPAN'

        transition = predict_transition(queue, stack, model_name, model, tree, vocab, max_edus, leaf_ind)

        if transition.action == 'shift':
            node = Node(relation='SPAN',
                        text=queue.pop(),
                        span=[leaf_ind, leaf_ind])
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

    return stack.pop()


def predict_transition(queue, stack, model_name, model, tree, vocab, max_edus, top_ind_in_queue):
    transition = Transition()
    sample = Sample()
    sample.state = gen_config(queue, stack, top_ind_in_queue)
    sample.tree = tree
    _, x_vecs = add_features_per_sample(sample, vocab, max_edus)
    action, alter_action = model.predict(np.array(x_vecs).reshape(1, -1))

    # correct invalid action
    if len(stack) < 2 and action != 'SHIFT':
        action = 'SHIFT'
    elif (not queue) and action == 'SHIFT':
        action = alter_action

    if action == 'SHIFT':
        transition.action = 'shift'	
    else:
        transition.action = 'reduce'

        splitaction = action.split('-')
        nuc = splitaction[1]
        rel = splitaction[2]

        if nuc == 'NS':
            transition.nuclearity.append('Nucleus')
            transition.nuclearity.append('Satellite')
        elif nuc == 'SN':
            transition.nuclearity.append('Satellite')
            transition.nuclearity.append('Nucleus')
        else:
            transition.nuclearity.append('Nucleus')
            transition.nuclearity.append('Nucleus')
        transition.relation = rel

    return transition


def gen_config(queue, stack, top_ind_in_queue):
    q_temp = []
    if queue:
        q_temp.append(top_ind_in_queue)
    return genstate(stack, q_temp)
