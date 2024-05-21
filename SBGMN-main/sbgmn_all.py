import os, sys, random, argparse, subprocess
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from model import SBGMN
from dataset import *
from sklearn.metrics import f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', default=BASE_DIR, help='Current Dir')
parser.add_argument('--device', type=str, default='cuda:1', help='Devices')

parser.add_argument('--dataset_name', type=str, default='review-1')
parser.add_argument('--model_type', type=str, default='SBGMN') # SBGMN

parser.add_argument('--a_emb_size', type=int, default=32, help='Embeding A Size')
parser.add_argument('--b_emb_size', type=int, default=32, help='Embeding B Size')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight Decay')
parser.add_argument('--lr', type=float, default=0.005, help='Learning Rate')   # 0.001
parser.add_argument('--seed', type=int, default=2024, help='Random seed')
parser.add_argument('--epoch', type=int, default=600, help='Epoch')

parser.add_argument('--gnn_layer_num', type=int, default=2, help='GNN Layer')  #2
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout')#0.6
parser.add_argument('--end_loss_rate', type=float, default=0.8, help='Loss rate')   #0.8
parser.add_argument('--gnn_loss_rate', type=float, default=0.4, help='GNNs loss rate')   #0.4
parser.add_argument('--view_hidden', type=int, default=64, help='view hiedden')   # review 16
parser.add_argument('--view_relate_rate', type=float, default=0.4, help='view relate rate')#0.4
parser.add_argument('--gnn_kl_rate', type=float, default=0.8, help='gnn kl loss rate') #0.8

args = parser.parse_args()

# TODO

exclude_hyper_params = ['dirpath', 'device']
hyper_params = dict(vars(args))
for exclude_p in exclude_hyper_params:
    del hyper_params[exclude_p]

hyper_params = "~".join([f"{k}-{v}" for k, v in hyper_params.items()])

from torch.utils.tensorboard import SummaryWriter
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# setup seed
# setup_seed(args.seed)
# args.device = 'cpu'
args.device = torch.device(args.device)

@torch.no_grad()
def test_and_val(pred_y, y, mode='val', epoch=0):
    preds = pred_y.cpu().numpy()
    y = y.cpu().numpy()

    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    test_y = y

    auc = roc_auc_score(test_y, preds)
    f1 = f1_score(test_y, preds)
    macro_f1 = f1_score(test_y, preds, average='macro')
    micro_f1 = f1_score(test_y, preds, average='micro')
    pos_ratio = np.sum(test_y) / len(test_y)
    res = {
        f'{mode}_auc': auc,
        f'{mode}_f1': f1,
        f'{mode}_pos_ratio': pos_ratio,
        f'{mode}_epoch': epoch,
        f'{mode}_macro_f1': macro_f1,
        f'{mode}_micro_f1': micro_f1,
    }
    for k, v in res.items():
        mode, _, metric = k.partition('_')
    return res


def plot_curve(result_dir, x_list, y_list, mode,
               labels=('train', 'valid', 'test'), colors=('red', 'yellow', 'black'), show=False):
    if show:
        plt.cla()

    plt.figure(figsize=(8, 5))
    for x, y, label, color in zip(x_list, y_list, labels, colors):
        plt.plot(x, y, label=label, marker='o', linestyle='-', color=color)
    plt.title(mode + ' Curve Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(mode)
    plt.grid(True)
    plt.legend()
    if show:
        plt.show()
    plt.savefig(f'{result_dir}/{mode}-curve.png')

def run():
    train_edgelist, val_edgelist, test_edgelist = load_data(args.dataset_name)

    set_a_num, set_b_num = DATA_EMB_DIC[args.dataset_name]
    train_y = np.array([i[-1] for i in train_edgelist])
    val_y = np.array([i[-1] for i in val_edgelist])
    test_y = np.array([i[-1] for i in test_edgelist])

    train_y = torch.from_numpy((train_y + 1) / 2).float().to(args.device)
    val_y = torch.from_numpy((val_y + 1) / 2).float().to(args.device)
    test_y = torch.from_numpy((test_y + 1) / 2).float().to(args.device)
    # get edge lists
    edgelists = load_edgelists(train_edgelist, args.dataset_name)

    edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg, \
    edgelist_pos, edgelist_neg = edgelists

    model = eval(args.model_type)(edgelists,
                  dataset_name=args.dataset_name,
                  layer_num=args.gnn_layer_num,
                  view_hidden=args.view_hidden,
                  emb_size_a=args.a_emb_size,
                  emb_size_b=args.b_emb_size,
                  dropout=args.dropout,
                  view_relate_rate=args.view_relate_rate,
                  device=args.device
                                  )
    model = model.to(args.device)

    # print(model.train())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-8)

    res_best = {'val_auc': 0}
    # res_best = {'val_auc': 0, 'val_f1': 0}

    if args.model_type != 'SBGMNOnlyView' and args.model_type != 'SBGMNOnlyView_WL':
        all_nodes = list(set(edgelist_pos.keys()).union(*edgelist_pos.values()))
        # 创建邻接矩阵
        adjacency_matrix1 = np.zeros((len(all_nodes), len(all_nodes)))
        # 填充邻接矩阵
        for start_node, end_nodes in edgelist_pos.items():
            for end_node in end_nodes:
                start_index = all_nodes.index(start_node)
                end_index = all_nodes.index(end_node)
                adjacency_matrix1[start_index, end_index] = 1
        adjacency_matrix1 = torch.tensor(adjacency_matrix1).float().to(args.device)
        adjacency_matrix1 = torch.matmul(adjacency_matrix1, adjacency_matrix1)
        adjacency_matrix1.fill_diagonal_(0)

    auc_trainvalues, auc_valvalues, auc_testvalues = [], [], []
    f1_trainvalues, f1_valvalues, f1_testvalues = [], [], []
    macro_trainvalues, macro_valvalues, macro_testvalues = [], [], []
    micro_trainvalues, micro_valvalues, micro_testvalues = [], [], []

    for epoch in tqdm(range(1, args.epoch + 1)):
        # train
        model.train()
        optimizer.zero_grad()
        pred_y, embedding_a, embedding_b, embedding_a2, embedding_b2 = model(train_edgelist)

        if args.model_type == 'SBGMNOnlyView' or args.model_type == 'SBGMNOnlyView_WL':
            loss = model.loss(pred_y, train_y)
        else:
            loss1 = model.loss(pred_y, train_y)
            loss3 = model.multi_gnn_loss(
                embedding_a, embedding_b, embedding_a2, embedding_b2, adjacency_matrix1, args.gnn_loss_rate)
            end_loss_rate = args.end_loss_rate
            loss = end_loss_rate * loss1 + (1 - end_loss_rate) * loss3
        # loss = loss1
        loss.backward()
        optimizer.step()
        print('loss', loss, 'lr', scheduler.get_last_lr())
        scheduler.step()

        res_cur = {}
        # if epoch % 5 == 0:
        if True:
            # val/test
            model.eval()
            pred_y, embedding_a, embedding_b, embedding_a2, embedding_b2 = model(train_edgelist)
            res = test_and_val(pred_y, train_y, mode='train', epoch=epoch)
            res_cur.update(res)
            auc_trainvalues.append(res_cur['train_auc'])
            f1_trainvalues.append(res_cur['train_f1'])
            macro_trainvalues.append(res_cur['train_macro_f1'])
            micro_trainvalues.append(res_cur['train_micro_f1'])

            pred_val_y, embedding_a, embedding_b, embedding_a2, embedding_b2 = model(val_edgelist)
            res = test_and_val(pred_val_y, val_y, mode='val', epoch=epoch)
            res_cur.update(res)
            auc_valvalues.append(res_cur['val_auc'])
            f1_valvalues.append(res_cur['val_f1'])
            macro_valvalues.append(res_cur['val_macro_f1'])
            micro_valvalues.append(res_cur['val_micro_f1'])

            pred_test_y, embedding_a, embedding_b, embedding_a2, embedding_b2  = model(test_edgelist)
            res = test_and_val(pred_test_y, test_y, mode='test', epoch=epoch)
            res_cur.update(res)
            auc_testvalues.append(res_cur['test_auc'])
            f1_testvalues.append(res_cur['test_f1'])
            macro_testvalues.append(res_cur['test_macro_f1'])
            micro_testvalues.append(res_cur['test_micro_f1'])

            if res_cur['val_auc'] > res_best['val_auc']:
                res_best = res_cur
            print(res_cur)

    print('Done! Best Results:')
    print(res_best)
    print_list = ['test_auc', 'test_f1', 'test_macro_f1', 'test_micro_f1']
    for i in print_list:
        print(i, res_best[i], end=' ')

def main():
    # setup_seed(args.seed)
    print(" ".join(sys.argv))
    this_fpath = os.path.abspath(__file__)
    t = subprocess.run(f'cat {this_fpath}', shell=True, stdout=subprocess.PIPE)
    print(str(t.stdout, 'utf-8'))
    print('=' * 20)
    run()


if __name__ == "__main__":
    main()
