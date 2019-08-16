from nltk import tokenize
from collections import defaultdict
from utils import map_to_cluster
import re
import copy
import glob
import nltk
import os
import shutil


# debugging
print_sents = True
sents_dir = 'sents'


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


def preprocess(path, dis_files_dir, ser_files_dir='', bin_files_dir=''):
    trees, max_edus = binarize_files(path, dis_files_dir, bin_files_dir)
    if ser_files_dir != '':
        print_serial_files(path, trees, ser_files_dir)
    gen_sentences(trees, path, dis_files_dir)
    for tree in trees:
        sent_ind = 1
        fn = build_infile_name(tree._fname, path, dis_files_dir, ["out.edus", "edus"])
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


def binarize_files(base_path, dis_files_dir, bin_files_dir):
    trees = []
    max_edus = 0
    path = base_path / dis_files_dir
    assert os.path.isdir(path), "Path to dataset does not exist: " + dis_files_dir

    path = path / "*.dis"
    for fn in glob.glob(str(path)):
        tree = binarize_file(fn, bin_files_dir)
        trees.append(tree)
        if tree._root._span[1] > max_edus:
            max_edus = tree._root._span[1]
    return trees, max_edus


def binarize_file(infn, bin_files_dir):
    stack = []
    with open(infn, "r") as ifh: # .dis file
        lines = ifh.readlines()
        root = build_tree(lines[::-1], stack)

    binarize_tree(root)

    if bin_files_dir != '':
        outfn = infn.split(os.sep)[0]
        outfn += os.sep
        outfn += bin_files_dir
        outfn += os.sep
        outfn += extract_base_name_file(infn)
        outfn += ".out.dis"
        with open(outfn, "w") as ofh:
            print_dis_file(ofh, root, 0)

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

    # print("{}".format(line))
 
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
    line.replace("\\TT_ERR", '')
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


def print_dis_file(ofh, node, level):
    nuc = node._nuclearity
    rel = node._relation
    beg = node._span[0]
    end = node._span[1]
    if node._type == "leaf":
        # Nucleus (leaf 1) (rel2par span) (text _!Wall Street is just about ready to line_!) )
        print_spaces(ofh, level)
        text = node._text
        ofh.write("( {} (leaf {}) (rel2par {}) (text _!{}_!) )\n".format(nuc, beg, rel, text))
    else:
        if node._type == "Root":
            # ( Root (span 1 91)
            ofh.write("( Root (span {} {})\n".format(beg, end))
        else:
            # ( Nucleus (span 1 69) (rel2par Contrast)
            print_spaces(ofh, level)
            ofh.write("( {} (span {} {}) (rel2par {})\n".format(nuc, beg, end, rel))
        l = node._childs[0]
        r = node._childs[1]
        print_dis_file(ofh, l, level + 1)
        print_dis_file(ofh, r, level + 1) 
        print_spaces(ofh, level)
        ofh.write(")\n")

def print_spaces(ofh, level):
    n_spaces = 2 * level
    for i in range(n_spaces):
        ofh.write(" ")

# print serial tree files

def print_serial_files(base_path, trees, outdir):
    path = create_dir(base_path, outdir)

    for tree in trees:
        outfn = path
        outfn += os.sep
        outfn += tree._fname
        with open(outfn, "w") as ofh:
            print_serial_file(ofh, tree._root)

def print_serial_file(ofh, node, doMap=True):
    if node._type != "Root":
        nuc = node._nuclearity
        if doMap == True:
            rel = map_to_cluster(node._relation)
        else:
            rel = node._relation
        beg = node._span[0]
        end = node._span[1]
        ofh.write("{} {} {} {}\n".format(beg, end, nuc[0], rel))

    if node._type != "leaf":
        l = node._childs[0]
        r = node._childs[1]
        print_serial_file(ofh, l, doMap)
        print_serial_file(ofh, r, doMap)

def print_trees_stats(trees):
    rel_freq = defaultdict(int)

    for tree in trees:
        gen_tree_stats(tree._root, rel_freq)

    total = 0
    for _, v in rel_freq.items():
        total += v

    for k, v in rel_freq.items():
        rel_freq[k] = v / total

    rel_freq_list = [(k,v) for k, v in rel_freq.items()]

    rel_freq_list = sorted(rel_freq_list, key=lambda elem: elem[1])
    rel_freq_list = rel_freq_list[::-1]
    print("most frequent relations: {}".format(rel_freq_list[0:5]))

def gen_tree_stats(node, rel_freq):
    if node._type != "Root":
        nuc = node._nuclearity
        rel = map_to_cluster(node._relation)
        rel_freq[rel] += 1

    if node._type != "leaf":
        l = node._childs[0]
        r = node._childs[1]
        gen_tree_stats(l, rel_freq)
        gen_tree_stats(r, rel_freq)

def gen_sentences(trees, base_path, infiles_dir):
    if print_sents:
        if not os.path.isdir(sents_dir):
               os.makedirs(sents_dir)

    for tree in trees:
        fn = tree._fname
        fn = build_infile_name(tree._fname, base_path, infiles_dir, ["out", ""]) 
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

        if print_sents:
            fn_sents = build_file_name(tree._fname, base_path, sents_dir, "out.sents")
            with open(fn_sents, "w") as ofh:
                for sent in tree._sents[1:]:
                    ofh.write("{}\n".format(sent))


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


def build_infile_name(fname, base_path, dis_files_dir, suffs):
    for suf in suffs:
        fn = build_file_name(fname, base_path, dis_files_dir, suf)
        if os.path.exists(fn):
            return fn
    raise Exception("Invalid file path:" + fn)


def build_file_name(base_fn, base_path, files_dir, suf):
    fn = base_path / files_dir
    if suf != '':
        fn = fn / (base_fn + "." + suf)
    else:
        fn = fn / base_fn
    return str(fn)


def create_dir(base_path, outdir):
    path = base_path / outdir
    if path.exists():
        shutil.rmtree(str(base_path / outdir))
    path = str(path)
    os.makedirs(path)
    return path