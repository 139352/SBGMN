import os
import numpy as np
from collections import defaultdict
from common import DATA_EMB_DIC

# =========== function
def load_data(dataset_name):
    train_file_path = os.path.join('datasets', f'{dataset_name}_training.txt')
    val_file_path = os.path.join('datasets', f'{dataset_name}_val.txt')
    test_file_path = os.path.join('datasets', f'{dataset_name}_test.txt')


    train_edgelist = []
    with open(train_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            train_edgelist.append((a, b, s))

    val_edgelist = []
    with open(val_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            val_edgelist.append((a, b, s))

    test_edgelist = []
    with open(test_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            test_edgelist.append((a, b, s))

    return np.array(train_edgelist), np.array(val_edgelist), np.array(test_edgelist)


# ============= load data
def load_edgelists(edge_lists, dataset_name):
    edgelist_a_b_pos, edgelist_a_b_neg = defaultdict(list), defaultdict(list)
    edgelist_b_a_pos, edgelist_b_a_neg = defaultdict(list), defaultdict(list)
    edgelist_pos, edgelist_neg = defaultdict(list), defaultdict(list)
    edgelist_a_a_pos, edgelist_a_a_neg = defaultdict(list), defaultdict(list)
    edgelist_b_b_pos, edgelist_b_b_neg = defaultdict(list), defaultdict(list)

    set_a_num, set_b_num = DATA_EMB_DIC[dataset_name]
    for a, b, s in edge_lists:
        if s == 1:
            edgelist_a_b_pos[a].append(b)
            edgelist_b_a_pos[b].append(a)
            edgelist_pos[a].append(b+set_a_num)
            edgelist_pos[b+set_a_num].append(a)
        elif s == -1:
            edgelist_a_b_neg[a].append(b)
            edgelist_b_a_neg[b].append(a)
            edgelist_neg[a].append(b+set_a_num)
            edgelist_neg[b+set_a_num].append(a)
        else:
            # print(a, b, s)
            raise Exception("s must be -1/1")

    edge_list_a_a = defaultdict(lambda: defaultdict(int))
    edge_list_b_b = defaultdict(lambda: defaultdict(int))
    for a, b, s in edge_lists:
        for b2 in edgelist_a_b_pos[a]:
            edge_list_b_b[b][b2] += 1 * s
        for b2 in edgelist_a_b_neg[a]:
            edge_list_b_b[b][b2] -= 1 * s
        for a2 in edgelist_b_a_pos[b]:
            edge_list_a_a[a][a2] += 1 * s
        for a2 in edgelist_b_a_neg[b]:
            edge_list_a_a[a][a2] -= 1 * s

    for a1 in edge_list_a_a:
        for a2 in edge_list_a_a[a1]:
            v = edge_list_a_a[a1][a2]
            if a1 == a2: continue
            if v > 0:
                edgelist_a_a_pos[a1].append(a2)
            elif v < 0:
                edgelist_a_a_neg[a1].append(a2)

    for b1 in edge_list_b_b:
        for b2 in edge_list_b_b[b1]:
            v = edge_list_b_b[b1][b2]
            if b1 == b2: continue
            if v > 0:
                edgelist_b_b_pos[b1].append(b2)
            elif v < 0:
                edgelist_b_b_neg[b1].append(b2)

    return edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg, \
           edgelist_pos, edgelist_neg
