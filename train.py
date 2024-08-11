# coding=utf-8
import argparse
import os
import random
import sys
from datetime import datetime
from collections import defaultdict
parent_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir_path)
from pathlib import Path
import time
import logging
import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.seed import seed_everything
from concurrent.futures import ThreadPoolExecutor
from load_data import load_aig_data
from model import PolarGate
import warnings

# warnings.filterwarnings("ignore")


def get_logger(name, logfile=None):
    logger = logging.getLogger(name)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PolarGate_processed')
    parser.add_argument('--task_type', type=str, default='prob', choices=['prob', 'tt'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='PolarGate')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--in_dim', type=int, default=3)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--split_file', type=str, default='0.05-0.05-0.9')
    parser.add_argument('--feature_type', type=str, default='one-hot',
                        choices=['deepgate', 'spectral', 'one-hot'])
    parser.add_argument('--loss_type', type=str, default='mae', choices=['mae', 'mse'])
    parser.add_argument('--name_others', type=str, default='')

    args = parser.parse_args()

    args.data_root_path = Path.home().joinpath('AIGDataset', args.dataset)
    args.pi_edges_path = args.data_root_path.joinpath('npz', 'pi_edges.npz')
    args.tt_pair_path = args.data_root_path.joinpath('npz', 'labels.npz')

    os.makedirs(Path.cwd().joinpath('ft_saved'), exist_ok=True)
    os.makedirs(Path.cwd().joinpath('results'), exist_ok=True)

    if args.name_others != '':
        args.ft_model_path = Path.cwd().joinpath('ft_saved',
                f'{args.task_type}_{args.dataset}_{args.model}_{str(args.layer_num)}_{args.name_others}_state_dict.pth')
    else:
        args.ft_model_path = Path.cwd().joinpath('ft_saved',
                f'{args.task_type}_{args.dataset}_{args.model}_{str(args.layer_num)}_state_dict.pth')
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    return args


def remap_tensor(raw_data, data_dir):
    import json
    id_map_file = Path(data_dir).joinpath("processed", "node_id_map.json")
    with open(id_map_file, "r") as file:
        id_dict = json.load(file)
    new_idx = list(map(int, list(id_dict.keys())))
    new_data = raw_data[new_idx]
    return new_data


def remap_edges(edge_index, data_dir):
    import json
    id_map_file = Path(data_dir).joinpath("processed", "node_id_map.json")
    with open(id_map_file, "r") as file:
        id_dict = json.load(file)
    new_edge_index = edge_index.clone()
    for old_pos, new_pos in id_dict.items():
        old_pos = int(old_pos)
        new_pos = int(new_pos)
        new_edge_index[edge_index == old_pos] = new_pos
    return new_edge_index


def one_hot(idx, length):
    if type(idx) is int:
        idx = torch.LongTensor([idx]).unsqueeze(0)
    else:
        idx = torch.LongTensor(idx).unsqueeze(0).t()
    x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
    return x


def construct_node_feature(x, num_gate_types=3):
    gate_list = x[:, 1]
    gate_list = np.float32(gate_list)
    x_torch = one_hot(gate_list, num_gate_types)
    return x_torch


def zero_normalization(x):
    if x.shape[0] == 1: return x
    mean_x = torch.mean(x)
    std_x = torch.std(x) + 1e-8
    z_x = (x - mean_x) / std_x
    return z_x


def load_data_signed_parallel(args, graph_dirs, pi_edges_dict, tt_pair_dict):
    total_data = []

    def process_data(data_dir):
        graph_name = os.path.basename(data_dir)
        print('Parsing Graph: {}'.format(graph_name))
        data = load_aig_data(dataset=args.dataset, root=data_dir, train_size=0.8, val_size=0.1,
                                     test_size=0.1, data_split=1).to(args.device)
        data.to_unweighted()
        if args.feature_type == 'one-hot':
            node_features = np.genfromtxt(os.path.join(data_dir, 'raw/node-feat.csv'), delimiter=',')
            node_features_tensor_d = torch.from_numpy(node_features).float()
            node_features_tensor_o = construct_node_feature(node_features_tensor_d).to(args.device)
            node_features_tensor = remap_tensor(node_features_tensor_o, data_dir)
            assert node_features_tensor.shape[0] == data.num_nodes
        edge_index = data.edge_index.t()  # [num_edges, 2]
        edge_sign = data.edge_weight.long()
        edge_index_s = torch.cat([edge_index, edge_sign.unsqueeze(-1)], dim=-1)  # [num_edges, 3]

        pi_edges_list = list(pi_edges_dict[graph_name].values())
        pi_edges_signed = np.concatenate(pi_edges_list, axis=0)
        pi_edges_signed_tensor = torch.from_numpy(pi_edges_signed).long()
        pi_edges_weight = pi_edges_signed_tensor[:, 2].unsqueeze(1)
        new_pi_edges = remap_edges(pi_edges_signed_tensor[:, :2], data_dir)
        pi_edges_signed_tensor = torch.cat([new_pi_edges, pi_edges_weight], dim=1).to(args.device)

        tt_pair_index = tt_pair_dict[graph_name]['tt_pair_index']
        tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long, device=args.device)
        tt_pair_index_tensor = remap_edges(tt_pair_index, data_dir).t().contiguous()
        tt_dis_tensor = torch.tensor(tt_pair_dict[graph_name]['tt_dis'], dtype=torch.float32, device=args.device)

        node_labels = np.genfromtxt(os.path.join(data_dir, 'raw/prob.csv'), delimiter=None)
        node_labels_tensor = torch.from_numpy(node_labels).float().to(args.device)
        node_labels_tensor = remap_tensor(node_labels_tensor, data_dir)

        total_data.append([data_dir, edge_index_s, pi_edges_signed_tensor, tt_pair_index_tensor,
                           node_features_tensor, node_labels_tensor, tt_dis_tensor])

    with ThreadPoolExecutor(args.num_workers) as executor:
        executor.map(process_data, graph_dirs)

    return total_data


def parse_data_parallel(args):
    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'train.txt')) as file:
        lines = file.readlines()
    train_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]
    random.shuffle(train_file)

    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'valid.txt')) as file:
        lines = file.readlines()
    valid_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]
    random.shuffle(valid_file)

    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'test.txt')) as file:
        lines = file.readlines()
    test_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]
    random.shuffle(test_file)

    pi_edges_dict = np.load(args.pi_edges_path, allow_pickle=True)['pi_edges'].item()

    tt_pair_dict = np.load(args.tt_pair_path, allow_pickle=True)['labels'].item()
    train_data = load_data_signed_parallel(args, train_file, pi_edges_dict, tt_pair_dict)
    valid_data = load_data_signed_parallel(args, valid_file, pi_edges_dict, tt_pair_dict)
    test_data = load_data_signed_parallel(args, test_file, pi_edges_dict, tt_pair_dict)

    return train_data, valid_data, test_data


def load_model(args):
    in_dim = args.in_dim
    out_dim = args.out_dim
    model = PolarGate(args=args, node_num=0, device=args.device, in_dim=in_dim, out_dim=out_dim,
                     layer_num=args.layer_num, lamb=5).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def AIG_Not_Edge_Loss(edge_index_s, out, typ='mae'):
    not_edges = edge_index_s[edge_index_s[:, 2] == -1]
    input_out = out[not_edges[:, 0]]
    output_out = out[not_edges[:, 1]]
    edges_sum = input_out + output_out
    if typ == 'mae':
        mae = F.l1_loss(edges_sum, torch.ones_like(edges_sum))
    else:
        mae = F.mse_loss(edges_sum, torch.ones_like(edges_sum))
    return mae


def AIG_AND_Edge_Loss(edge_index_s, out, typ='mae'):
    sorted_indices = torch.argsort(edge_index_s[:, 1])
    edge_index_s = edge_index_s[sorted_indices]
    and_edges = edge_index_s[edge_index_s[:, 2] == 1]
    inputs = out[and_edges[:, 0]]
    outputs = out[and_edges[:, 1]]
    inputs = inputs.view(-1, 2)
    outputs = outputs.view(-1, 2)
    min_inputs = torch.min(inputs, dim=1)[0]
    if typ == 'mae':
        loss = F.l1_loss(outputs[:, 0], min_inputs)
    else:
        loss = F.mse_loss(outputs[:, 0], min_inputs)
    return loss


def AIG_PI_AND_Edge_Loss(edge_index_s, out, typ='mae'):
    edge_index = edge_index_s[:, :2]
    edge_sign = edge_index_s[:, 2]
    neg_nodes = edge_index[edge_sign == -1][:, 1]

    all_nodes = torch.arange(out.size(0), device=out.device)
    pos_nodes = torch.tensor(list(set(all_nodes.tolist()) - set(neg_nodes.tolist())), device=out.device)
    loss_nodes = out[pos_nodes]

    if typ == 'mae':
        loss = F.l1_loss(loss_nodes, torch.zeros_like(loss_nodes))
    else:
        loss = F.mse_loss(loss_nodes, torch.zeros_like(loss_nodes))

    return loss


def test(args, model, test_data):
    model.eval()
    results = {'prob': 0.0, 'not': 0.0, 'and': 0.0, 'tt': 0.0, 'pi_and': 0.0,
               'level': defaultdict(lambda: {'value': 0.0, 'cnt': 0.0})}
    with torch.no_grad():
        for i, [data_dir, edge_index_s, pi_edges_signed_tensor, tt_pair_index_tensor,
                node_features_tensor, node_labels_tensor, tt_dis_tensor] in enumerate(test_data):
            node_labels_tensor = node_labels_tensor.unsqueeze(1)

            out_emb, out = model(node_features_tensor, edge_index_s)

            node_a = out_emb[tt_pair_index_tensor[0]]
            node_b = out_emb[tt_pair_index_tensor[1]]
            emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
            emb_dis_z = zero_normalization(emb_dis)
            tt_dis_z = zero_normalization(tt_dis_tensor)

            results['not'] += AIG_Not_Edge_Loss(edge_index_s, out, typ=args.loss_type)
            results['and'] += AIG_AND_Edge_Loss(edge_index_s, out, typ=args.loss_type)
            results['pi_and'] += AIG_PI_AND_Edge_Loss(edge_index_s, out, typ=args.loss_type)

            if args.loss_type == 'mae':
                prob_loss = F.l1_loss(out, node_labels_tensor)
                func_loss = F.l1_loss(emb_dis_z, tt_dis_z)
            else:
                prob_loss = F.mse_loss(out, node_labels_tensor)
                func_loss = F.mse_loss(emb_dis_z, tt_dis_z)

            results['prob'] += prob_loss.item()
            results['tt'] += func_loss.item()

    results['prob'] /= len(test_data)
    results['not'] /= len(test_data)
    results['and'] /= len(test_data)
    results['tt'] /= len(test_data)
    results['pi_and'] /= len(test_data)

    return results


def train(args, model, optimizer, train_data, valid_data, logger):
    logger.info('*********** Start End-to-End Train ***********')

    best_loss = 999999999999
    patience = args.patience
    total_time = 0.0
    cnt_epoch = 0

    for epoch in range(1, 1 + args.epochs):
        t = time.time()
        model.train()
        total_prob_loss = 0
        total_tt_loss = 0
        random.shuffle(train_data)
        for i, [data_dir, edge_index_s, pi_edges_signed_tensor, tt_pair_index_tensor,
                           node_features_tensor, node_labels_tensor, tt_dis_tensor] in enumerate(train_data):
            node_labels_tensor = node_labels_tensor.unsqueeze(1)
            out_emb, out = model(node_features_tensor, edge_index_s)

            node_a = out_emb[tt_pair_index_tensor[0]]
            node_b = out_emb[tt_pair_index_tensor[1]]
            emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
            emb_dis_z = zero_normalization(emb_dis)
            tt_dis_z = zero_normalization(tt_dis_tensor)

            if args.loss_type == 'mae':
                prob_loss = F.l1_loss(out, node_labels_tensor)
                func_loss = F.l1_loss(emb_dis_z, tt_dis_z)
            else:
                prob_loss = F.mse_loss(out, node_labels_tensor)
                func_loss = F.mse_loss(emb_dis_z, tt_dis_z)

            total_prob_loss += prob_loss.item()
            total_tt_loss += func_loss.item()

            if args.task_type == 'prob':
                prob_loss.backward()
            elif args.task_type == 'tt':
                func_loss.backward()

            if (i + 1) % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()

        total_time += time.time() - t
        cnt_epoch += 1
        valid_results = test(args, model, valid_data)

        total_prob_loss /= len(train_data)
        total_tt_loss /= len(train_data)

        logger.info('Epoch: {:02d} | [Train] Prob: {:.4f} Func: {:.4f} |'
                    '[Valid] Prob: {:.4f} Func: {:.4f} | PI_AND: {:.4f} AND: {:.4f} NOT: {:.4f}'.format(
                    epoch, total_prob_loss, total_tt_loss, valid_results['prob'], valid_results['tt'],
                    valid_results['pi_and'], valid_results['and'], valid_results['not']))

        if valid_results[args.task_type] < best_loss:
            best_loss = valid_results[args.task_type]
            best_model_info = {
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())
            }
            patience = args.patience

        patience -= 1
        if patience <= 0:
            break
    torch.save(best_model_info, args.ft_model_path)

    return total_time / cnt_epoch


if __name__ == '__main__':
    args = parameter_parser()
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    if args.name_others == '':
        logname = f'{args.task_type}_{args.dataset}_{args.model}_{args.layer_num}__{args.split_file}_{timestamp}.log'
    else:
        logname = f'{args.task_type}_{args.dataset}_{args.model}_{args.layer_num}__{args.split_file}_{args.name_others}_{timestamp}.log'
    logfile = os.path.join('results', logname)
    logger = get_logger(__name__, logfile=logfile)

    # logger.info(dict(args._get_kwargs()))
    # logger.info(f'Path.home():{Path.home()}')

    train_data, valid_data, test_data = parse_data_parallel(args)

    model, optimizer = load_model(args)

    avg_train_time = train(args, model, optimizer, train_data, valid_data, logger)
    del model

    model, optimizer = load_model(args)
    checkpoint = torch.load(args.ft_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = test(args, model, test_data)

    logger.info('*********** Test Result ***********')
    logger.info('[Test] Prob: {:.4f} Func: {:.4f} | PI_AND: {:.4f} AND: {:.4f} NOT: {:.4f} | Avg.Train_time: {:.4f}'.format(
        test_results['prob'], test_results['tt'], test_results['pi_and'], test_results['and'], test_results['not'],
        avg_train_time))

