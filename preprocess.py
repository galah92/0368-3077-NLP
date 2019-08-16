from nltk import tokenize
from collections import defaultdict
from utils import map_to_cluster
import re
import copy
import glob
import nltk
import os
import shutil


class Node():

    def __init__(self, _nuclearity='', _relation='', _childs=None, _type='', _span=[0, 0], _text=''):
        self._nuclearity = _nuclearity
        self._relation = _relation
        self._childs = [] if _childs is None else _childs
        self._type = _type
        self._span = _span
        self._text = _text

    def copy(self):
        to = Node()
        to._nuclearity = self._nuclearity
        to._relation = self._relation
        to._childs = copy.copy(self._childs)
        to._span = copy.copy(self._span)
        to._text = self._text
        to._type = self._type
        return to


class TreeInfo():

    def __init__(self):
        self._fname = ''
        self._root = ''
        self._EDUS_table = ['']
        self._sents = ['']
        self._edu_to_sent_ind = ['']
        self._edu_word_tag_table = [['']]


def preprocess(dis_files_dir, ser_files_dir=''):
    trees, max_edus = binarize_files(dis_files_dir)
    if ser_files_dir != '':
        print_serial_files(trees, ser_files_dir)
    gen_sentences(trees, dis_files_dir)
    for tree in trees:
        sent_ind = 1
        fn = build_infile_name(tree._fname, dis_files_dir, ["out.edus", "edus"])
        with open(fn) as f:
            for edu in f:
                edu = edu.strip()
                edu_tokenized = tokenize.word_tokenize(edu)
                tree._edu_word_tag_table.append(nltk.pos_tag(edu_tokenized))
                tree._EDUS_table.append(edu)
                if not is_edu_in_sent(edu, tree._sents[sent_ind]):
                    sent_ind += 1
                tree._edu_to_sent_ind.append(sent_ind)
    return trees, max_edus


def binarize_files(dis_files_dir):
    trees = []
    max_edus = 0
    for fn in glob.glob(str(dis_files_dir / '*.dis')):
        tree = binarize_file(fn)
        trees.append(tree)
        if tree._root._span[1] > max_edus:
            max_edus = tree._root._span[1]
    return trees, max_edus


def binarize_file(infn):
    stack = []
    with open(infn, "r") as ifh:  # .dis file
        lines = [line.split('//')[0] for line in ifh.readlines()]
        root = build_tree(lines[::-1], stack)

    binarize_tree(root)

    tree_info = TreeInfo()
    tree_info._root = root
    tree_info._fname = extract_base_name_file(infn)
    return tree_info


def extract_base_name_file(fn):
    base_name = fn.split(os.sep)[-1]
    base_name = base_name.split('.')[0]
    return base_name

# lines are the content of .dis" file


def build_tree(lines, stack):
    line = lines.pop(-1)
    line = line.strip()
    node = Node()

    # ( Root (span 1 54)
    m = re.match("\( Root \(span (\d+) (\d+)\)", line)
    if m:
        tokens = m.groups()
        node._type = "Root"
        node._span = [int(tokens[0]), int(tokens[1])]
        stack.append(node)
        return build_tree_childs_iter(lines, stack)

    # ( Nucleus (span 1 34) (rel2par Topic-Drift)
    m = re.match("\( (\w+) \(span (\d+) (\d+)\) \(rel2par ([\w-]+)\)", line)
    if m:
        tokens = m.groups()
        node._nuclearity = tokens[0]
        node._type = "span"
        node._span = [int(tokens[1]), int(tokens[2])]
        node._relation = tokens[3]
        stack.append(node)
        return build_tree_childs_iter(lines, stack)

    # ( Satellite (leaf 3) (rel2par attribution) (text _!Southern Co. 's Gulf Power Co. unit_!) )
    m = re.match("\( (\w+) \(leaf (\d+)\) \(rel2par ([\w-]+)\) \(text (.+)", line)
    tokens = m.groups()
    node._type = "leaf"
    node._nuclearity = tokens[0]
    node._span = [int(tokens[1]), int(tokens[1])] 
    node._relation = tokens[2]
    text = tokens[3]
    text = text[2:]
    text = text[:-5]
    node._text = text
    return node


def build_tree_childs_iter(lines, stack):
    while True:
        line = lines[-1]
        line.strip()
        words = line.split()
        if words[0] == ")":
            lines.pop(-1)
            break

        node = build_tree(lines, stack)
        stack[-1]._childs.append(node)
    return stack.pop(-1)


def binarize_tree(node):
    if not node._childs:
        return

    if len(node._childs) > 2:
        stack = []
        for child in node._childs:
            stack.append(child)

        node._childs = []
        while len(stack) > 2:
            r = stack.pop(-1)
            l = stack.pop(-1)

            t = l.copy()
            t._childs = [l, r]
            t._span = [l._span[0], r._span[1]]
            t._type = "span"
            stack.append(t)
        r = stack.pop(-1)
        l = stack.pop(-1)
        node._childs = [l, r]
    else:
        l = node._childs[0]
        r = node._childs[1]

    binarize_tree(l)
    binarize_tree(r)


def print_serial_files(trees, outdir):
    create_dir(outdir)
    for tree in trees:
        with (outdir / tree._fname).open('w') as ofh:
            print_serial_file(ofh, tree._root)


def print_serial_file(ofh, node, do_map=True):
    if node._type != "Root":
        nuc = node._nuclearity
        if do_map:
            rel = map_to_cluster(node._relation)
        else:
            rel = node._relation
        beg = node._span[0]
        end = node._span[1]
        ofh.write("{} {} {} {}\n".format(beg, end, nuc[0], rel))

    if node._type != "leaf":
        l = node._childs[0]
        r = node._childs[1]
        print_serial_file(ofh, l, do_map)
        print_serial_file(ofh, r, do_map)


def gen_sentences(trees, infiles_dir):
    for tree in trees:
        fn = tree._fname
        fn = build_infile_name(tree._fname, infiles_dir, ["out", ""]) 
        with open(fn) as fh:
            content = ''
            lines = fh.readlines()
            for line in lines:
                line = sent_transform(line)
                content += line 
            content = content.replace(' \n', ' ')
            content = content.replace('\n', ' ')
            content = content.replace('  ', ' ')
            sents = tokenize.sent_tokenize(content)
            for sent in sents:
                if sent.strip() == '':
                    continue
                tree._sents.append(sent)


def is_edu_in_sent(edu, sent):
    edu1 = sent_transform(edu)
    return edu1 in sent


def sent_transform(string):
    string = string.replace(' . . .', '')
    string = string.replace('Mr.', 'Mr')
    string = string.replace('No.', 'No')
    # 'and. . . some'
    string = re.sub('([^.]*)\. \. \. ([^.]+)', r'\1 \2', string)
    string = re.sub('([a-zA-Z])\.([a-zA-Z])\.([a-zA-Z])\.', r'\1\2\3', string)
    string = re.sub('([a-zA-Z])\.([a-zA-Z])\.', r'\1\2', string)
    string = re.sub('([A-Z][a-z]+)\.', r'\1', string)
    return string


def build_infile_name(fname, dis_files_dir, suffs):
    for suf in suffs:
        fn = build_file_name(fname, dis_files_dir, suf)
        if os.path.exists(fn):
            return fn
    raise Exception("Invalid file path:" + fn)


def build_file_name(base_fn, files_dir, suf):
    fn = files_dir
    if suf != '':
        fn = fn / (base_fn + "." + suf)
    else:
        fn = fn / base_fn
    return str(fn)


def create_dir(path):
    if path.exists():
        shutil.rmtree(str(path))
    os.makedirs(str(path))
    return path
