from collections import deque
from preprocess import Node, print_serial_file
from evaluation import eval as evaluate
from features import add_features_per_sample
from train_data import Sample, genstate
from neural_network import neural_net_predict
from relations_inventory import ind_toaction_map
import numpy as np
import torch
from tqdm import tqdm


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


def parse_files(model_name, model, trees, vocab, tag_to_idx, infiles_dir, gold_files_dir, pred_outdir):
    max_edus = max(tree._root.span[1] for tree in trees)
    pred_outdir.mkdir(exist_ok=True)
    for tree in tqdm(trees):
        tree_file = list(infiles_dir.glob(f'{tree._fname}*.edus'))[0]
        queue = deque(line.strip() for line in tree_file.open())
        stack = deque()
        root = parse_file(queue, stack, model_name, model, tree, vocab, max_edus, tag_to_idx)
        print_serial_file(pred_outdir / tree._fname, root)
    evaluate(gold_files_dir, pred_outdir)


def parse_file(queue, stack, model_name, model, tree, vocab, max_edus, tag_to_idx):
    ## RNN ##
    # samples, _ = gen_train_data([tree])
    # x_vecs, _, sents_idx = get_features([tree], samples, vocab, tag_to_idx)
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

        transition = predict_transition(queue, stack, model_name, model, tree, vocab, max_edus, tag_to_idx, leaf_ind)

        if transition.action == "shift":
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


def predict_transition(queue, stack, model_name, model, tree, vocab, max_edus, tag_to_idx, top_ind_in_queue):
    transition = Transition()
    sample = Sample()
    sample.state = gen_config(queue, stack, top_ind_in_queue)
    sample.tree = tree
    _, x_vecs = add_features_per_sample(sample, vocab, max_edus, tag_to_idx)

    if model_name == "rnn":
        alter_action = 'REDUCE-NN-ELABORATION'  # DEBUG ONLY
    elif model_name == "neural":
        pred = neural_net_predict(model, x_vecs)
        action = ind_toaction_map[pred.argmax()]
        _, indices = torch.sort(pred)
        alter_action = ind_toaction_map[indices[-2]]
    elif model_name == 'multi_label':
        clf1, clf2, clf3 = model

        if hasattr(clf1, "decision_function"):
            pred1 = clf1.decision_function(np.array(x_vecs).reshape(1, -1))
        else:
            pred1 = clf1.predict_proba(np.array(x_vecs).reshape(1, -1))

        if hasattr(clf2, "decision_function"):
            pred2 = clf2.decision_function(np.array(x_vecs).reshape(1, -1))
        else:
            pred2 = clf2.predict_proba(np.array(x_vecs).reshape(1, -1))

        if hasattr(clf3, "decision_function"):
            pred3 = clf3.decision_function(np.array(x_vecs).reshape(1, -1))
        else:
            pred3 = clf3.predict_proba(np.array(x_vecs).reshape(1, -1))

        a1 = 'REDUCE'
        # fix the action if needed
        a2 = clf2.classes_[np.argmax(pred2)] if clf2.classes_[np.argmax(pred2)] != 'SHIFT' else clf2.classes_[np.argsort(pred2).squeeze()[-2]] 
        a3 = clf3.classes_[np.argmax(pred3)] if clf3.classes_[np.argmax(pred3)] != 'SHIFT' else clf3.classes_[np.argsort(pred3).squeeze()[-2]]

        if clf1.classes_[np.argmax(pred1)] == 'SHIFT':
            action = 'SHIFT'
            alter_action = ('-').join(a1, a2, a3)
        else:
            action = ('-').join(a1, a2, a3)

    else:
        if hasattr(model, "decision_function"):
            pred = model.decision_function(np.array(x_vecs).reshape(1,-1))
        else:
            pred = model.predict_proba(np.array(x_vecs).reshape(1,-1))
        action = ind_toaction_map[model.classes_[np.argmax(pred)]]
        alter_action = ind_toaction_map[model.classes_[np.argsort(pred).squeeze()[-2]]]

    # correct invalid action
    if len(stack) < 2 and action != "SHIFT":
        action = "SHIFT"
    elif (not queue) and action == "SHIFT":
        action = alter_action

    if action == "SHIFT":
        transition.action = "shift"	
    else:
        transition.action = "reduce"

        splitaction = action.split("-")
        nuc = splitaction[1]
        rel = splitaction[2]

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
    return genstate(stack, q_temp)
