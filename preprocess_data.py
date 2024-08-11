from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from rich.progress import Progress
from rich.progress import track

from typing import Optional, Callable, List

import shutil
import copy
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
"""how to install deepgate: https://github.com/zshi0616/python-deepgate/tree/main"""
import deepgate
from deepgate.utils.data_utils import read_npz_file
from deepgate.utils.aiger_utils import aig_to_xdata
from deepgate.utils.circuit_utils import get_fanin_fanout, read_file, add_node_index, feature_gen_connect
from deepgate.parser_func import parse_pyg_mlpgate


class NpzParser():
    '''
        Parse the npz file into an inmemory torch_geometric.data.Data object
        modified by jiawei liu.
    '''

    def __init__(self, data_dir, circuit_path, label_path,
                 random_shuffle=True, trainval_split=None):   # add test data (ljw 2024.4.8)
        if trainval_split is None:
            trainval_split = [0.2, 0.1, 0.1]
        self.data_dir = data_dir
        dataset = self.inmemory_dataset(data_dir, circuit_path, label_path)
        if random_shuffle:
            torch.manual_seed(0)
            perm = torch.randperm(len(dataset))
            dataset = dataset[perm]
        data_len = len(dataset)
        training_cutoff = int(data_len * trainval_split[0])
        valid_cutoff = int(data_len * trainval_split[1])
        self.train_dataset = dataset[:training_cutoff]
        self.val_dataset = dataset[training_cutoff:training_cutoff+valid_cutoff]
        self.test_dataset = dataset[training_cutoff+valid_cutoff:]

    def get_dataset(self):
        return self.train_dataset, self.val_dataset, self.test_dataset

    class inmemory_dataset(InMemoryDataset):
        def __init__(self, root, circuit_path, label_path, transform=None, pre_transform=None, pre_filter=None):
            self.name = 'npz_inmm_dataset'
            self.root = root
            self.circuit_path = circuit_path
            self.label_path = label_path
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            name = 'inmemory'
            return osp.join(self.root, name)

        @property
        def raw_file_names(self) -> List[str]:
            return [self.circuit_path, self.label_path]

        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass

        def process(self):
            data_list = []
            tot_pairs = 0
            circuits = read_npz_file(self.circuit_path)['circuits'].item()
            labels = read_npz_file(self.label_path)['labels'].item()

            for cir_idx, cir_name in enumerate(circuits):
                print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits),
                                                                      cir_idx / len(circuits) * 100))
                x = circuits[cir_name]["x"]
                signed_edge_file = self.root.parent.joinpath(cir_name, 'raw', 'signed_edge.csv')
                edge_index_s = pd.read_csv(signed_edge_file, header=None).values
                edge_index = edge_index_s
                tt_dis = labels[cir_name]['tt_dis']
                min_tt_dis = labels[cir_name]['min_tt_dis']
                tt_pair_index = labels[cir_name]['tt_pair_index']
                prob = labels[cir_name]['prob']

                rc_pair_index = labels[cir_name]['rc_pair_index']
                is_rc = labels[cir_name]['is_rc']

                if len(tt_pair_index) == 0 or len(rc_pair_index) == 0:
                    print('No tt or rc pairs: ', cir_name)
                    continue

                tot_pairs += len(tt_dis)

                graph = parse_pyg_mlpgate(
                    x, edge_index, tt_dis, min_tt_dis, tt_pair_index,
                    prob, rc_pair_index, is_rc, edge_index_s
                )
                graph.name = cir_name

                data_list.append(graph)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))

        def __repr__(self) -> str:
            return f'{self.name}({len(self)})'



def parse_bench_v2(bench_file):
    input_nodes, output_nodes, and_nodes, not_nodes = [], [], [], []
    edges = []

    # 读取文件
    with open(bench_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith('INPUT'):
                input_nodes.append(int(re.search(r'\((\d+)\)', line).group(1)))
            elif line.startswith('OUTPUT'):
                output_nodes.append(int(re.search(r'\((\d+)\)', line).group(1)))
            elif '=' in line and 'AND' in line:
                left, right = line.split('=')
                output_id = int(left.strip())
                inputs = re.findall(r'\((\d+), (\d+)\)', right)
                input1, input2 = map(int, inputs[0])
                edges.append([input1, output_id, 1])
                edges.append([input2, output_id, 1])
                and_nodes.append(output_id)
            elif '=' in line and 'NOT' in line:
                left, right = line.split('=')
                output_id = int(left.strip())
                input_id = int(re.search(r'\((\d+)\)', right).group(1))
                edges.append([input_id, output_id, -1])
                not_nodes.append(output_id)

    # 将连接关系存入Pandas DataFrame
    signed_edge = pd.DataFrame(edges, columns=['Input', 'Output', 'NOT_Flag'])

    id_map = {
        'input_nodes': input_nodes,
        'output_nodes': output_nodes,
        'and_nodes': and_nodes,
        'not_nodes': not_nodes
    }
    return id_map, signed_edge

def load_raw_data_and_transform(root_dir, circuit_path, label_path, bench_path):
    """ generate node-feat.csv, signed_edge.csv and prob.csv"""
    circuits = np.load(circuit_path, allow_pickle=True)['circuits'].item()
    labels = np.load(label_path, allow_pickle=True)['labels'].item()
    with Progress() as progress:
        task1 = progress.add_task("[red]Processing...", total=len(circuits))
        for cir_idx, cir_name in enumerate(circuits):
            bench_file = bench_path.joinpath(f'{cir_name}.bench')
            id_map, signed_edge = parse_bench_v2(bench_file)
            save_path = root_dir.joinpath(cir_name, 'raw')
            os.makedirs(save_path, exist_ok=True)
            x = circuits[cir_name]["x"]
            np.savetxt(save_path.joinpath('node-feat.csv'), x, delimiter=',', fmt='%d')
            signed_edge.to_csv(save_path.joinpath('signed_edge.csv'), header=False,  index=False)
            # with open(save_path.joinpath('id_map.pickle'), 'wb') as file:
            #     pickle.dump(id_map, file)
            prob = labels[cir_name]['prob']  # task 1 label

            np.savetxt(save_path.joinpath('prob.csv'), prob, delimiter=',', fmt='%s')
            # num_node = x.shape[0]
            # num_edge = signed_edge.shape[0]
            # with open(save_path.joinpath('num-node-list.csv'), 'w') as file:
            #     file.write(str(num_node) + '\n')
            # with open(save_path.joinpath('num-edge-list.csv'), 'w') as file:
            #     file.write(str(num_edge) + '\n')
            progress.update(task1, advance=1)


def generate_pi_edges(root_dir, circuit_path, label_path, bench_path):
    """ generate pi_edges.npz """
    circuits = np.load(circuit_path, allow_pickle=True)['circuits'].item()
    dataset = deepgate.NpzParser.inmemory_dataset(root_dir.joinpath('npz'), circuit_path, label_path)
    pi_edges = {}
    for cir_idx in track(range(len(dataset))):
        cir_name = dataset[cir_idx].name
        forward_level = dataset[cir_idx].forward_level
        forward_index = dataset[cir_idx].forward_index
        max_level = max(forward_level)
        pi_edges[cir_name] = {}
        bench_file = bench_path.joinpath(f'{cir_name}.bench')
        save_path = root_dir.joinpath(cir_name, 'raw')
        os.makedirs(save_path, exist_ok=True)
        id_map, signed_edge = parse_bench_v2(bench_file)

        pi_ids = torch.tensor(id_map['input_nodes'])
        node_num = circuits[cir_name]["x"].shape[0]
        edge_tensor = torch.tensor(signed_edge.values)

        rows = edge_tensor[:, 0]
        cols = edge_tensor[:, 1]
        weights = edge_tensor[:, 2].float()
        sparse_adj = torch.sparse.FloatTensor(
            indices=torch.stack([rows, cols]),
            values=weights,
            size=(node_num, node_num)
        )
        tmp_adj = sparse_adj.coalesce()

        for level in range(1, max_level):
            level_ids = forward_index[(forward_level == level).nonzero(as_tuple=True)[0]]
            if level > 1:
                tmp_adj = torch.sparse.mm(tmp_adj, sparse_adj)
            edges = tmp_adj.indices().t()
            weights = tmp_adj.values()
            edge_index = torch.cat((edges, weights.unsqueeze(1)), dim=1)
            mask1 = torch.isin(edge_index[:, 0], pi_ids)
            mask2 = torch.isin(edge_index[:, 1], level_ids)
            combined_mask = mask1 & mask2
            pi_edges[cir_name][level] = edge_index[combined_mask]
    np.savez_compressed(root_dir.joinpath('npz', 'pi_edges.npz'), pi_edges=pi_edges)


def save_split_data(root_dir, split_ratio):
    dataset = NpzParser(root_dir.joinpath('npz'), circuit_path, label_path,
                                 random_shuffle=True, trainval_split=split_ratio)  # dataset is a list of graphs
    train_dataset, val_dataset, test_dataset = dataset.get_dataset()
    train_valid_str = f"{split_ratio[0]}-{split_ratio[1]}-{split_ratio[2]}"   # train-valid-test
    save_dir = root_dir.joinpath('split', train_valid_str)
    os.makedirs(save_dir, exist_ok=True)

    train_data_name = [graph.name for graph in train_dataset]
    valid_data_name = [graph.name for graph in val_dataset]
    test_data_name = [graph.name for graph in test_dataset]
    print(len(train_data_name+valid_data_name+test_data_name))
    with open(str(save_dir.joinpath('train.txt')), "w") as file:
        for string in train_data_name:
            file.write(string + "\n")
    with open(str(save_dir.joinpath('valid.txt')), "w") as file:
        for string in valid_data_name:
            file.write(string + "\n")
    with open(str(save_dir.joinpath('test.txt')), "w") as file:
        for string in test_data_name:
            file.write(string + "\n")
    print('Save finished!')


if __name__ == '__main__':
    root_dir = Path.home().joinpath('AIGDataset', 'PolarGate_raw')
    circuit_path = root_dir.joinpath('npz', 'graphs.npz')
    label_path = root_dir.joinpath('npz', 'labels.npz')
    bench_path = root_dir.joinpath('rawaig')
    split_ratio = [0.01, 0.01, 0.98]

    load_raw_data_and_transform(root_dir, circuit_path, label_path, bench_path)
    generate_pi_edges(root_dir, circuit_path, label_path, bench_path)
    save_split_data(root_dir, split_ratio)
