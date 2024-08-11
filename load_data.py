import json
import shutil, os
from torch_geometric.data import (InMemoryDataset, download_url, Data)
from torch_geometric.typing import OptTensor, Tuple, Union
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix, is_undirected
from torch import FloatTensor, LongTensor
from typing import Optional, Callable, Union, List, Tuple
from torch_geometric.data import Data
import torch
import numpy as np


class AIG_data(InMemoryDataset):
    def __init__(self, name: str,  root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.root = root

        super().__init__(root, transform, pre_transform)
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> str:
        return 'signed_edge.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data = []
        edge_weight = []
        edge_index = []
        node_map = {}
        with open(self.raw_paths[0], 'r') as f:
            # next(f)
            for line in f:
                x = line.strip().split(',')
                assert len(x) == 3
                a, b = x[0], x[1]
                if a not in node_map:
                    node_map[a] = len(node_map)
                if b not in node_map:
                    node_map[b] = len(node_map)
                a, b = node_map[a], node_map[b]
                data.append([a, b])
                edge_weight.append(float(x[2]))

            edge_index = [[i[0], int(i[1])] for i in data]
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            edge_weight = torch.FloatTensor(edge_weight)
        os.makedirs(self.processed_dir, exist_ok=True)
        map_file = os.path.join(self.processed_dir, 'node_id_map.json')
        with open(map_file, 'w') as f:
            f.write(json.dumps(node_map))

        data = Data(edge_index=edge_index, edge_weight=edge_weight)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1


def node_class_split(data: Data,
                     train_size: Union[int, float] = None, val_size: Union[int, float] = None,
                     test_size: Union[int, float] = None, seed_size: Union[int, float] = None,
                     train_size_per_class: Union[int, float] = None, val_size_per_class: Union[int, float] = None,
                     test_size_per_class: Union[int, float] = None, seed_size_per_class: Union[int, float] = None,
                     seed: List[int] = [], data_split: int = 10) -> Data:
    r""" Train/Val/Test/Seed split for node classification tasks.
    The size parameters can either be int or float.
    If a size parameter is int, then this means the actual number, if it is float, then this means a ratio.
    Arg types:
        * **data** (Data or DirectedData, required) - The data object for data split.
        * **train_size** (int or float, optional) - The size of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **val_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size** (int or float, optional) - The size of random splits for the seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **train_size_per_class** (int or float, optional) - The size per class of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **val_size_per_class** (int or float, optional) - The size per class of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size_per_class** (int or float, optional) - The size per class of random splits for the testing dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size_per_class** (int or float, optional) - The size per class of random splits for seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **seed** (An empty list or a list with the length of data_split, optional) - The random seed list for each data split.
        * **data_split** (int, optional) - number of splits (Default : 10)

    Return types:
        * **data** (Data or DirectedData) - The data object includes train_mask, val_mask and test_mask.
    """
    if train_size is None and train_size_per_class is None:
        raise ValueError(
            'Please input the values of train_size or train_size_per_class!')

    if seed_size is not None and seed_size_per_class is not None:
        raise Warning(
            'The seed_size_per_class will be considered if both seed_size and seed_size_per_class are given!')
    if test_size is not None and test_size_per_class is not None:
        raise Warning(
            'The test_size_per_class will be considered if both test_size and test_size_per_class are given!')
    if val_size is not None and val_size_per_class is not None:
        raise Warning(
            'The val_size_per_class will be considered if both val_size and val_size_per_class are given!')
    if train_size is not None and train_size_per_class is not None:
        raise Warning(
            'The train_size_per_class will be considered if both train_size and val_size_per_class are given!')

    if len(seed) == 0:
        seed = list(range(data_split))
    if len(seed) != data_split:
        raise ValueError(
            'Please input the random seed list with the same length of {}!'.format(data_split))

    if isinstance(data.y, torch.Tensor):
        labels = data.y.numpy()
    else:
        if train_size_per_class is None and val_size_per_class is None and test_size_per_class is None:
            labels = np.array([i for i in range(data.num_nodes)])
        else:
            labels = np.array(data.y)
    masks = {}
    masks['train'], masks['val'], masks['test'], masks['seed'] = [], [], [], []
    for i in range(data_split):
        random_state = np.random.RandomState(seed[i])
        train_indices, val_indices, test_indices, seed_indices = get_train_val_test_seed_split(random_state,
                                                                                               labels, train_size_per_class, val_size_per_class, test_size_per_class, seed_size_per_class,
                                                                                               train_size, val_size, test_size, seed_size)

        train_mask = np.zeros((labels.shape[0], 1), dtype=int)
        train_mask[train_indices, 0] = 1
        val_mask = np.zeros((labels.shape[0], 1), dtype=int)
        val_mask[val_indices, 0] = 1
        test_mask = np.zeros((labels.shape[0], 1), dtype=int)
        test_mask[test_indices, 0] = 1
        seed_mask = np.zeros((labels.shape[0], 1), dtype=int)
        if len(seed_indices) > 0:
            seed_mask[seed_indices, 0] = 1

        mask = {}
        mask['train'] = torch.from_numpy(train_mask).bool()
        mask['val'] = torch.from_numpy(val_mask).bool()
        mask['test'] = torch.from_numpy(test_mask).bool()
        mask['seed'] = torch.from_numpy(seed_mask).bool()

        masks['train'].append(mask['train'])
        masks['val'].append(mask['val'])
        masks['test'].append(mask['test'])
        masks['seed'].append(mask['seed'])

    data.train_mask = torch.cat(masks['train'], axis=-1)
    data.val_mask = torch.cat(masks['val'], axis=-1)
    data.test_mask = torch.cat(masks['test'], axis=-1)
    data.seed_mask = torch.cat(masks['seed'], axis=-1)
    return data


def sample_per_class(random_state: np.random.RandomState, labels: List[int], num_examples_per_class: Union[int, float],
                     forbidden_indices: Optional[List[int]] = None, force_indices: Optional[List[int]] = None) -> List[int]:
    r"""This function is modified from https://github.com/flyingtango/DiGCN/blob/main/code/Citation.py. It samples a set of nodes per class.
    If num_exmples_per_class is int, then this means the actual number, if it is float, then this means a ratio.

    Arg types:
        * **random_state** (np.random.RandomState) - Numpy random state for random selection.
        * **labels** (List[int]) - Node labels array.
        * **num_examples_per_class** (int or float) - Number of nodes per class.
        * **forbidden_indices** (List[int]) - Nodes to be avoided when selection.
        * **force_indices** (List[int]) - Node list to be selected.

    Return types:
        * **selection** (List) - A list of node indices to be selected.
    """
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if ((forbidden_indices is None or sample_index not in forbidden_indices)
                        and (force_indices is None or sample_index in force_indices)):
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    if isinstance(num_examples_per_class, int):
        return np.concatenate(
            [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
             for class_index in range(num_classes)
             ])
    elif isinstance(num_examples_per_class, float):
        selection = []
        if force_indices is None:
            values, counts = np.unique(labels, return_counts=True)
        else:
            values, counts = np.unique(
                labels[force_indices], return_counts=True)
        for class_index, count in zip(values, counts):
            size = int(num_examples_per_class*count)
            selection.extend(random_state.choice(
                sample_indices_per_class[class_index], size, replace=False))
        return selection
    else:
        raise TypeError(
            "Please input a float or int number for the parameter num_examples_per_class.")


def get_train_val_test_seed_split(random_state: np.random.RandomState,
                                  labels: List[int],
                                  train_size_per_class: Union[int, float] = None, val_size_per_class: Union[int, float] = None,
                                  test_size_per_class: Union[int, float] = None, seed_size_per_class: Union[int, float] = None,
                                  train_size: Union[int, float] = None, val_size: Union[int, float] = None,
                                  test_size: Union[int, float] = None, seed_size: Union[int, float] = None) -> Tuple[List[int], List[int], List[int], List[int]]:
    r"""Get train/validation/test/seed splits based on the input setting.
    Arg types:
        * **random_state** (np.random.RandomState): Numpy random state for random selection.
        * **train_size** (int ,optional): The size of random splits for the training dataset.
        * **val_size** (int, optional): The size of random splits for the validation dataset.
        * **test_size** (int, optional): The size of random splits for the validation dataset. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size** (int or float, optional): The size of random splits for the seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **train_size_per_class** (int or float, optional): The size per class of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **val_size_per_class** (int or float, optional): The size per class of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size_per_class** (int or float, optional): The size per class of random splits for the testing dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
                    (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size_per_class** (int or float, optional): The size per class of random splits for seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.

    Return types:
        * **train_indices** (List) - A List includes the node indices for training.
        * **val_indices** (List) - A List includes the node indices for validation.
        * **test_indices** (List) - A List includes the node indices for testing.
        * **seed_indices** (List) - A list includes the node indices for seed nodes (could be empty).
    """
    num_samples = labels.shape[0]
    remaining_indices = list(range(num_samples))

    if train_size is None and train_size_per_class is None:
        raise ValueError(
            'Please input the values of train_size or train_size_per_class!')

    if seed_size is not None and seed_size_per_class is not None:
        raise Warning(
            'The seed_size_per_class will be considered if both seed_size and seed_size_per_class are given!')
    if test_size is not None and test_size_per_class is not None:
        raise Warning(
            'The test_size_per_class will be considered if both test_size and test_size_per_class are given!')
    if val_size is not None and val_size_per_class is not None:
        raise Warning(
            'The val_size_per_class will be considered if both val_size and val_size_per_class are given!')
    if train_size is not None and train_size_per_class is not None:
        raise Warning(
            'The train_size_per_class will be considered if both train_size and val_size_per_class are given!')

    if train_size_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_size_per_class)
    else:
        # select train examples with no respect to class distribution
        if isinstance(train_size, int):
            train_indices = random_state.choice(
                remaining_indices, train_size, replace=False)
        elif isinstance(train_size, float):
            train_indices = random_state.choice(remaining_indices, int(
                train_size*len(remaining_indices)), replace=False)
        else:
            raise TypeError(
                "Please input a float or int number for the parameter train_size.")

    if seed_size_per_class is not None:
        seed_indices = sample_per_class(
            random_state, labels, seed_size_per_class, force_indices=train_indices)
    elif seed_size is not None:
        # select train examples with no respect to class distribution
        if isinstance(seed_size, int):
            seed_indices = random_state.choice(
                train_indices, seed_size, replace=False)
        elif isinstance(seed_size, float):
            seed_indices = random_state.choice(train_indices, int(
                seed_size*len(train_indices)), replace=False)
        else:
            raise TypeError(
                "Please input a float or int number for the parameter seed_size.")
    else:
        seed_indices = []

    val_indices = []
    if val_size_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_size_per_class, forbidden_indices=train_indices)
        forbidden_indices = np.concatenate((train_indices, val_indices))
    elif val_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        if isinstance(val_size, int):
            val_indices = random_state.choice(
                remaining_indices, val_size, replace=False)
        elif isinstance(val_size, float):
            val_indices = random_state.choice(remaining_indices, int(
                val_size*len(remaining_indices)), replace=False)
        else:
            raise TypeError(
                "Please input a float or int number for the parameter val_size.")
        forbidden_indices = np.concatenate((train_indices, val_indices))
    else:
        forbidden_indices = train_indices

    if test_size_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_size_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        if isinstance(test_size, int):
            test_indices = random_state.choice(
                remaining_indices, test_size, replace=False)
        elif isinstance(test_size, float):
            test_indices = random_state.choice(remaining_indices, int(
                test_size*len(remaining_indices)), replace=False)
        else:
            raise TypeError(
                "Please input a float or int number for the parameter test_size.")
    else:  # use all the rest as test set
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    assert len(set(seed_indices)) == len(seed_indices)
    # assert training, validation and test sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_size_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_size_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_size_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_size_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices, seed_indices


class SignedData(Data):
    r"""
    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        edge_weight (Tensor, optional): Edge weights with shape
            :obj:`[num_edges,]`. (default: :obj:`None`)
        y (Tensor, optional): Graph-level or node-level ground-truth labels
            with arbitrary shape. (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        A (sp.spmatrix or a tuple of sp.spmatrix, optional): SciPy sparse adjacency matrix,
            or a tuple of the positive and negative parts. (default: :obj:`None`)
        init_data (Data, optional): Initial data object, whose attributes will be inherited. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """

    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, edge_weight: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None,
                 A: Union[Tuple[sp.spmatrix, sp.spmatrix], sp.spmatrix, None] = None,
                 init_data: Optional[Data] = None, **kwargs):
        super().__init__(x=x, edge_index=edge_index,
                         edge_attr=edge_attr, y=y,
                         pos=pos, **kwargs)
        if A is None:
            A = to_scipy_sparse_matrix(edge_index, edge_weight)
        elif isinstance(A, tuple):
            A_p_scipy = A[0]
            A_n_scipy = A[1]
            A = A_p_scipy - A_n_scipy

        self.A = A.tocoo()
        self.edge_weight = FloatTensor(self.A.data)
        self.edge_index = LongTensor(np.array(self.A.nonzero()))
        self.num_nodes = self.A.shape[0]
        if init_data is not None:
            self.inherit_attributes(init_data)

    def separate_positive_negative(self):
        ind = self.edge_weight > 0
        self.edge_index_p = self.edge_index[:, ind]
        self.edge_weight_p = self.edge_weight[ind]
        ind = self.edge_weight < 0
        self.edge_index_n = self.edge_index[:, ind]
        self.edge_weight_n = - self.edge_weight[ind]
        self.A_p = to_scipy_sparse_matrix(
            self.edge_index_p, self.edge_weight_p, num_nodes=self.num_nodes)
        self.A_n = to_scipy_sparse_matrix(
            self.edge_index_n, self.edge_weight_n, num_nodes=self.num_nodes)

    def clear_separate_attributes(self):
        for name in ['edge_index_p', 'edge_index_n', 'edge_weight_p', 'edge_weight_n', 'A_p', 'A_n']:
            delattr(self, name)

    @property
    def is_signed(self) -> bool:
        return bool(self.edge_weight.max()*self.edge_weight.min() < 0)

    @property
    def is_directed(self) -> bool:
        return not is_undirected(self.edge_index, self.edge_weight)

    @property
    def is_weighted(self) -> bool:
        self.separate_positive_negative()
        res = self.edge_weight_p.max() != self.edge_weight_p.min(
        ) or self.edge_weight_n.max() != self.edge_weight_n.min()
        self.clear_separate_attributes()
        return bool(res)

    def to_unweighted(self):
        if hasattr(self, 'edge_weight'):
            self.edge_weight = self.edge_weight.sign()
            self.A = to_scipy_sparse_matrix(self.edge_index, self.edge_weight)
        if hasattr(self, 'edge_weight_p'):
            self.separate_positive_negative()

    def inherit_attributes(self, data: Data):
        for k in data.to_dict().keys():
            if k not in self.to_dict().keys():
                setattr(self, k, getattr(data, k))

    def node_split(self, train_size: Union[int, float] = None, val_size: Union[int, float] = None,
                   test_size: Union[int, float] = None, seed_size: Union[int, float] = None,
                   train_size_per_class: Union[int, float] = None, val_size_per_class: Union[int, float] = None,
                   test_size_per_class: Union[int, float] = None, seed_size_per_class: Union[int, float] = None,
                   seed: List[int] = [], data_split: int = 2):
        r""" Train/Val/Test/Seed split for node classification tasks.
        The size parameters can either be int or float.
        If a size parameter is int, then this means the actual number, if it is float, then this means a ratio.
        ``train_size`` or ``train_size_per_class`` is mandatory, with the former regardless of class labels.
        Validation and seed masks are optional.
        If test_size and test_size_per_class are both None, all the remaining nodes after selecting training (and validation) nodes will be included.

        Args:
            data (torch_geometric.data.Data or DirectedData, required): The data object for data split.
            train_size (int or float, optional): The size of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
            val_size (int or float, optional): The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
            test_size (int or float, optional): The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
                        (Default: None. All nodes not selected for training/validation are used for testing)
            seed_size (int or float, optional): The size of random splits for the seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
            train_size_per_class (int or float, optional): The size per class of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
            val_size_per_class (int or float, optional): The size per class of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
            test_size_per_class (int or float, optional): The size per class of random splits for the testing dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
                        (Default: None. All nodes not selected for training/validation are used for testing)
            seed_size_per_class (int or float, optional): The size per class of random splits for seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
            seed (An empty list or a list with the length of data_split, optional): The random seed list for each data split.
            data_split (int, optional): number of splits (Default : 2)

        """
        self = node_class_split(self, train_size=train_size, val_size=val_size,
                                test_size=test_size, seed_size=seed_size, train_size_per_class=train_size_per_class,
                                val_size_per_class=val_size_per_class, test_size_per_class=test_size_per_class,
                                seed_size_per_class=seed_size_per_class, seed=seed, data_split=data_split)


def load_aig_data(dataset: str = 'deepgate', root: str = './tmp_data/',
                          transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                          train_size: Union[int, float] = None, val_size: Union[int, float] = None,
                          test_size: Union[int, float] = None, seed_size: Union[int, float] = None,
                          train_size_per_class: Union[int, float] = None, val_size_per_class: Union[int, float] = None,
                          test_size_per_class: Union[int, float] = None, seed_size_per_class: Union[int, float] = None,
                          seed: List[int] = [], data_split: int = 10) -> SignedData:

    data = AIG_data(
        name=dataset, root=root, transform=transform, pre_transform=pre_transform)[0]
    signed_dataset = SignedData(
        edge_index=data.edge_index, edge_weight=data.edge_weight, init_data=data)
    if train_size is not None or train_size_per_class is not None:
        signed_dataset.node_split(train_size=train_size, val_size=val_size,
                                  test_size=test_size, seed_size=seed_size, train_size_per_class=train_size_per_class,
                                  val_size_per_class=val_size_per_class, test_size_per_class=test_size_per_class,
                                  seed_size_per_class=seed_size_per_class, seed=seed, data_split=data_split)
    return signed_dataset
