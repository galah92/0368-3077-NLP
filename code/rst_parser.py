from collections import deque
from preprocess import Node, print_serial_file, extract_base_name_file, create_dir, build_infile_name
from evaluation import eval as evaluate
from features import add_features_per_sample
from train_data import Sample, gen_state
from model import neural_net_predict, linear_predict
from relations_inventory import ind_to_action_map
import numpy as np
import random
import torch
import os


class Transition():

    def __init__(self):
        self._nuclearity = [] # <nuc>, <nuc>
        self._relation = '' # cluster relation
        self._action = '' # shift or 'reduce'

    def gen_str(self):
        s = self._action
        if s != 'shift':
            s += '-' + ''.join(x[0] for x in self._nuclearity) + '-' + self._relation
        return s.upper()


def parse_files(base_path, model_name, model, trees, vocab, max_edus, y_all, tag_to_ind_map, baseline, infiles_dir, gold_files_dir, pred_outdir="pred"):
    path_to_out = create_dir(base_path, pred_outdir)

    for tree in trees: 
        fn = build_infile_name(tree._fname, base_path, infiles_dir, ["out.edus", "edus"])
        with open(fn) as fh:
            queue = deque(line.strip() for line in fh)
        stack = deque()
        root = parse_file(queue, stack, model_name, model, tree, vocab, max_edus, y_all, tag_to_ind_map, baseline)
        predfn = path_to_out
        predfn += os.sep
        predfn += tree._fname
        with open(predfn, "w") as ofh:
            print_serial_file(ofh, root, False)
    evaluate(gold_files_dir, "pred")


def parse_file(queue, stack, model_name, model, tree, vocab, max_edus, y_all, tag_to_ind_map, baseline):
    leaf_ind = 1
    while queue or len(stack) != 1:
        node = Node()
        node._relation = 'SPAN'

        if baseline:
            transition = most_freq_baseline(queue, stack)
        else:
            transition = predict_transition(queue, stack, model_name, model, tree, vocab, max_edus, y_all, tag_to_ind_map, leaf_ind)

        if transition._action == "shift":
            node = Node(_relation='SPAN',
                        _text=queue.pop(),
                        _type='leaf',
                        _span=[leaf_ind, leaf_ind])
            leaf_ind += 1
        else:
            r = stack.pop()
            l = stack.pop()
            node._childs.append(l)
            node._childs.append(r)
            l._nuclearity = transition._nuclearity[0]
            r._nuclearity = transition._nuclearity[1]
            if l._nuclearity == "Satellite":
                l._relation = transition._relation
            elif r._nuclearity == "Satellite":
                r._relation = transition._relation	
            else:
                l._relation = transition._relation
                r._relation = transition._relation

            if (not queue) and (not stack):
                node._type = "Root"
            else:
                node._type = "span"
            node._span = [l._span[0], r._span[1]]
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
            transition._nuclearity.append("Nucleus")
            transition._nuclearity.append("Satellite")
        elif nuc == "SN":
            transition._nuclearity.append("Satellite")
            transition._nuclearity.append("Nucleus")
        else:
            transition._nuclearity.append("Nucleus")
            transition._nuclearity.append("Nucleus")
        transition._relation = rel

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
        ind = random.randint(0,1)
        transition._action = actions[ind]
    else:
        transition._action = "reduce"
        
    if transition._action == "shift":
        return transition

    transition._relation = 'ELABORATION'
    transition._nuclearity.append("Nucleus")
    transition._nuclearity.append("Satellite")

    return transition
