from utils import map_to_cluster
from collections import deque
from tqdm import tqdm
import copy
import nltk
import re
from pytorch_pretrained_bert.tokenization import BertTokenizer


class Node():

    def __init__(self, nuclearity='', relation='', childs=None, span=[0, 0], text=''):
        self.nuclearity = nuclearity
        self.relation = relation
        self.childs = [] if childs is None else childs
        self.span = span
        self.text = text

    def binarize(self):
        if not self.childs:
            return
        if len(self.childs) > 2:
            stack = deque(self.childs)
            while len(stack) > 2:
                right = stack.pop()
                left = stack.pop()
                temp = copy.copy(left)
                temp.childs = [left, right]
                temp.span = [left.span[0], right.span[1]]
                stack.append(temp)
            right = stack.pop()
            left = stack.pop()
            self.childs = [left, right]
        else:
            left, right = self.childs
        left.binarize()
        right.binarize()

    def _postorder(self, order=None):
        if self.nuclearity == 'Root':
            order = deque()
        for child in self.childs:
            child._postorder(order)
        if self.nuclearity != 'Root':
            order.append(self)
        return order

    def to_file(self, file_path):
        with file_path.open('w') as ofh:
            ofh.writelines(f'{node.span[0]} {node.span[1]} {node.nuclearity[0]} {node.relation}\n'
                           for node in self._postorder())

    def get_edu_ind(self):
        if not self.childs:
            return self.span[0]
        left = self.childs[0]
        right = self.childs[1]
        if left.nuclearity == 'Nucleus':
            return left.get_edu_ind()
        return right.get_edu_ind()


class TreeInfo():

    def __init__(self, root=None, filename=None):
        self.filename = filename
        self.root = root
        self.edus = ['']
        self.sents_idx = ['']
        self.pos_tags = [['']]


def load_trees(dis_dir, tree_list_dir=None):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

    if tree_list_dir is not None:
        tree_list_dir.mkdir(exist_ok=True)

    trees = []
    max_edus = 0
    for tree_file in dis_dir.glob('*.edus*'):
        # populate sentences
        content = ''.join(sent_transform(line) for line in tree_file.open())
        content = content.replace(' \n', ' ').replace('\n', ' ').replace('  ', ' ')
        sentences = [''] + [sent for sent in nltk.tokenize.sent_tokenize(content) if sent.strip() != '']
        # populate all the rest
        max_edus = max(max_edus, sum(1 for _ in tree_file.open()))
        sent_ind = 1
        tree = TreeInfo(filename=tree_file.with_suffix(''))
        with tree_file.open() as f:
            for edu in f:
                edu = edu.strip()
                tree.pos_tags.append(nltk.pos_tag(tokenizer.tokenize(edu)))
                tree.edus.append(edu)
                if sent_transform(edu) not in sentences[sent_ind]:
                    sent_ind += 1
                tree.sents_idx.append(sent_ind)
        tree_dis_file = tree_file.with_suffix('.dis')
        if tree_dis_file.is_file():
            tree.root = binarize_file(tree_dis_file)
            # tree.root.to_file(tree.filename)
        trees.append(tree)
    return trees, max_edus


def binarize_file(dis_path):
    lines = [line.split('//')[0] for line in dis_path.open()]
    root = build_tree(lines[::-1])
    root.binarize()
    return root


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
        return build_root_childs(lines, stack)

    # ( Nucleus (span 1 34) (rel2par Topic-Drift)
    m = re.match("\( (\w+) \(span (\d+) (\d+)\) \(rel2par ([\w-]+)\)", line)
    if m:
        tokens = m.groups()
        node.nuclearity = tokens[0]
        node.span = [int(tokens[1]), int(tokens[2])]
        node.relation = map_to_cluster(tokens[3])
        stack.append(node)
        return build_root_childs(lines, stack)

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


def build_root_childs(lines, stack):
    while True:
        words = lines[-1].strip().split()
        if words[0] == ")":
            lines.pop(-1)
            break
        node = build_tree(lines, stack)
        stack[-1].childs.append(node)
    return stack.pop()


def sent_transform(string):
    string = string.replace(' . .', '')
    string = string.replace(' . . .', '')
    string = string.replace(' . . . .', '')
    string = string.replace('Mr.', 'Mr')
    string = string.replace('No.', 'No')
    # 'and. . . some'
    string = re.sub('([^.]*)\. \. \. ([^.]+)', r'\1 \2', string)
    string = re.sub('([a-zA-Z])\.([a-zA-Z])\.([a-zA-Z])\.', r'\1\2\3', string)
    string = re.sub('([a-zA-Z])\.([a-zA-Z])\.', r'\1\2', string)
    string = re.sub('([A-Z][a-z]+)\.', r'\1', string)
    return string
