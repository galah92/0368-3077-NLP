from collections import deque
from preprocess import Node, print_serial_file
from evaluation import eval as evaluate
from features import add_features_per_sample
from train_data import Sample, gen_state
from models import neural_net_predict, linear_predict
from relations_inventory import ind_to_action_map
import numpy as np
import random
import torch


class Transition():

    def __init__(self):
        self.nuclearity = []  # <nuc>, <nuc>
        self.relation = ''  # cluster relation
        self._action = ''  # shift or 'reduce'

    def gen_str(self):
        s = self._action
        if s != 'shift':
            s += '-' + ''.join(x[0] for x in self.nuclearity) + '-' + self.relation
        return s.upper()


def parse_files(model_name, model, trees, vocab, y_all, tag_to_ind_map, baseline, infiles_dir, gold_files_dir, pred_outdir):
    max_edus = max(tree._root.span[1] for tree in trees)
    pred_outdir.mkdir(exist_ok=True)
    for tree in trees:
        tree_file = list(infiles_dir.glob(f'{tree._fname}*.edus'))[0]
        queue = deque(line.strip() for line in tree_file.open())
        stack = deque()
        root = parse_file(queue, stack, model_name, model, tree, vocab, max_edus, y_all, tag_to_ind_map, baseline)
        print_serial_file(pred_outdir / tree._fname, root)
    evaluate(gold_files_dir, pred_outdir)


def parse_file(queue, stack, model_name, model, tree, vocab, max_edus, y_all, tag_to_ind_map, baseline):
    leaf_ind = 1
    while queue or len(stack) != 1:
        node = Node()
        node.relation = 'SPAN'

        if baseline:
            transition = most_freq_baseline(queue, stack)
        else:
            transition = predict_transition(queue, stack, model_name, model, tree, vocab, max_edus, y_all, tag_to_ind_map, leaf_ind)

        if transition._action == "shift":
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
            if l.nuclearity == "Satellite":
                l.relation = transition.relation
            elif r.nuclearity == "Satellite":
                r.relation = transition.relation
            else:
                l.relation = transition.relation
                r.relation = transition.relation

            if not queue and not stack:
                node.nuclearity = 'Root'
            node.span = [l.span[0], r.span[1]]
        stack.append(node)

    return stack.pop()


def predict_transition(queue, stack, model_name, model, tree, vocab, max_edus, y_all, tag_to_ind_map, top_ind_in_queue):
    transition = Transition()
    sample = Sample()
    sample._state = gen_config(queue, stack, top_ind_in_queue)
    sample._tree = tree
    _, x_vecs = add_features_per_sample(sample, vocab, max_edus, tag_to_ind_map)

    if model_name == "neural":
        pred = neural_net_predict(model, x_vecs)
        action = ind_to_action_map[pred.argmax()]
        _, indices = torch.sort(pred)
    else:
        pred = linear_predict(model, [x_vecs])
        action = ind_to_action_map[y_all[np.argmax(pred)]]
        indices = np.argsort(pred)	

    # correct invalid action
    if len(stack) < 2 and action != "SHIFT":
        action = "SHIFT"
    elif (not queue) and action == "SHIFT":
        action = ind_to_action_map[indices[-2]]

    if action == "SHIFT":
        transition._action = "shift"	
    else:
        transition._action = "reduce"

        split_action = action.split("-")
        nuc = split_action[1]
        rel = split_action[2]

        if nuc == "NS":
            transition.nuclearity.append("Nucleus")
            transition.nuclearity.append("Satellite")
        elif nuc == "SN":
            transition.nuclearity.append("Satellite")
            transition.nuclearity.append("Nucleus")
        else:
            transition.nuclearity.append("Nucleus")
            transition.nuclearity.append("Nucleus")
        transition.relation = rel

    return transition


def gen_config(queue, stack, top_ind_in_queue):
    q_temp = []
    if queue:
        q_temp.append(top_ind_in_queue)
    return gen_state(stack, q_temp)


def most_freq_baseline(queue, stack):
    transition = Transition()

    if len(stack) < 2:
        transition._action = "shift"
    elif queue:
        actions = ["shift", "reduce"]
        ind = random.randint(0, 1)
        transition._action = actions[ind]
    else:
        transition._action = "reduce"

    if transition._action == "shift":
        return transition

    transition.relation = 'ELABORATION'
    transition.nuclearity.append("Nucleus")
    transition.nuclearity.append("Satellite")

    return transition
