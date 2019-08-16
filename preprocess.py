from utils import map_to_cluster
from collections import deque
import shutil
import copy
import nltk
import re
import os


class Node():

    def __init__(self, nuclearity='', relation='', childs=None, span=[0, 0], text=''):
        self.nuclearity = nuclearity
        self.relation = relation
        self.childs = [] if childs is None else childs
        self.span = span
        self.text = text


class TreeInfo():

    def __init__(self):
        self._fname = ''
        self._root = ''
        self._EDUS_table = ['']
        self._sents = ['']
        self._edu_to_sent_ind = ['']
        self._edu_word_tag_table = [['']]


def preprocess(dis_dir, ser_files_dir=''):
    trees = [binarize_file(dis_file) for dis_file in dis_dir.glob('*.dis')]
    if ser_files_dir != '':
        print_serial_files(trees, ser_files_dir)
    gen_sentences(trees, dis_dir)
    for tree in trees:
        sent_ind = 1
        fn = build_infile_name(tree._fname, dis_dir, ["out.edus", "edus"])
        with open(fn) as f:
            for edu in f:
                edu = edu.strip()
                edu_tokenized = nltk.tokenize.word_tokenize(edu)
                tree._edu_word_tag_table.append(nltk.pos_tag(edu_tokenized))
                tree._EDUS_table.append(edu)
                if not is_edu_in_sent(edu, tree._sents[sent_ind]):
                    sent_ind += 1
                tree._edu_to_sent_ind.append(sent_ind)
    return trees


def binarize_file(dis_path):
    with dis_path.open('r') as dis:
        lines = [line.split('//')[0] for line in dis.readlines()]
        root = build_tree(lines[::-1])
    binarize_tree(root)
    tree_info = TreeInfo()
    tree_info._root = root
    tree_info._fname = extract_base_name_file(str(dis_path))
    return tree_info


def extract_base_name_file(fn):
    base_name = fn.split(os.sep)[-1]
    base_name = base_name.split('.')[0]
    return base_name


def build_tree(lines, stack=None):
    if stack is None:
        stack = deque()
    line = lines.pop(-1)
    line = line.strip()
    node = Node()

    # ( Root (span 1 54)
    m = re.match("\( Root \(span (\d+) (\d+)\)", line)
    if m:
        tokens = m.groups()
        node.nuclearity = 'Root'
        node.span = [int(tokens[0]), int(tokens[1])]
        stack.append(node)
        return build_treechilds_iter(lines, stack)

    # ( Nucleus (span 1 34) (rel2par Topic-Drift)
    m = re.match("\( (\w+) \(span (\d+) (\d+)\) \(rel2par ([\w-]+)\)", line)
    if m:
        tokens = m.groups()
        node.nuclearity = tokens[0]
        node.span = [int(tokens[1]), int(tokens[2])]
        node.relation = map_to_cluster(tokens[3])
        stack.append(node)
        return build_treechilds_iter(lines, stack)

    # ( Satellite (leaf 3) (rel2par attribution) (text _!Southern Co. 's Gulf Power Co. unit_!) )
    m = re.match("\( (\w+) \(leaf (\d+)\) \(rel2par ([\w-]+)\) \(text (.+)", line)
    tokens = m.groups()
    node.nuclearity = tokens[0]
    node.span = [int(tokens[1]), int(tokens[1])] 
    node.relation = map_to_cluster(tokens[2])
    text = tokens[3]
    text = text[2:]
    text = text[:-5]
    node.text = text
    return node


def build_treechilds_iter(lines, stack):
    while True:
        line = lines[-1]
        line.strip()
        words = line.split()
        if words[0] == ")":
            lines.pop(-1)
            break
        node = build_tree(lines, stack)
        stack[-1].childs.append(node)
    return stack.pop()


def binarize_tree(node):
    if not node.childs:
        return
    if len(node.childs) > 2:
        stack = deque(node.childs)
        while len(stack) > 2:
            right = stack.pop()
            left = stack.pop()
            temp = copy.copy(left)
            temp.childs = [left, right]
            temp.span = [left.span[0], right.span[1]]
            stack.append(temp)
        right = stack.pop()
        left = stack.pop()
        node.childs = [left, right]
    else:
        left, right = node.childs
    binarize_tree(left)
    binarize_tree(right)


def print_serial_files(trees, outdir):
    create_dir(outdir)
    for tree in trees:
        print_serial_file(outdir / tree._fname, tree._root)


def postorder(node, order=None):
    if node.nuclearity == 'Root':
        order = deque()
    for child in node.childs:
        postorder(child, order)
    if node.nuclearity != 'Root':
        order.append(node)
    return order


def print_serial_file(file_path, root):
    with file_path.open('w') as ofh:
        for node in postorder(root):
            ofh.write(f'{node.span[0]} {node.span[1]} {node.nuclearity[0]} {node.relation}\n')


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
            sents = nltk.tokenize.sent_tokenize(content)
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
