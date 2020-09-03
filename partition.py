import math
from typing import List
from typing import Dict
import warnings
import numpy as np

class decision_tree:
    epsilon = 0
    flag = False

    # initialize objects
    def __init__(self, attribute: List[List[int]]):
        self.partitions: Dict[str, List[int]] = dict()
        self.types = len(attribute[0])
        self.attribute = attribute
        self.rows = len(attribute)

    # read data from a dataset file
    def read_data_file(self,fname):
        data = []
        with open(fname) as file:
             rows_count = -1
             for line in file:
                data.append(line)
                rows_count += 1
        string = ''.join([str(elem) for elem in data])
        a1, a2, a3, attribute = np.loadtxt(string, delimiter=' ',
                                           usecols=(0, 1, 2, 3),
                                           unpack=True)
        return a1, a2, a3, attribute, rows_count

    # find the unique val(s) in each column
    def unique_vals(rows, col):
        return set([row[col] for row in rows])

    # read data from a partition file
    def read_partition_file(self, fname):
        with open(fname) as file:
            data = file.readlines()
        for line in data:
            if len(line.strip()):
                store = line.split()
                self.partitions[store[0]] = list(map(int, store[1:]))
        set_obj = set()
        for j, k in self.partitions.items():
            for i in k:
                set_obj.add(i)
        total = set(range(1, self.rows + 1))

    # compute the entropy from the target attribute
    def compute_by_target_attribute(self, rows: List[int]) -> float:
        entropy = 0
        attribute_types = list(self.attribute[i - 1][-1] for i in rows)
        unique_attribute_types = set(attribute_types)
        for v in unique_attribute_types:
            f = attribute_types.count(v) / len(attribute_types)
            entropy += -f * math.log2(f)
        return entropy

    # compute the entropy based on the given attribute(s)
    def get_attr_entropy(self, attr: int, rows: List[int]) -> float:
        entropy = 0
        attribute_types = list((self.attribute[i - 1][attr],
                                self.attribute[i - 1][-1]) for i in rows)
        unique_vals_attribute = set(i[0] for i in attribute_types)
        unique_vals_target = set(i[1] for i in attribute_types)
        for av in unique_vals_attribute:
            feature_entropy = 0
            d = list(i[0] for i in attribute_types).count(av)
            for tv in unique_vals_target:
                f = attribute_types.count((av, tv)) / (d + decision_tree.epsilon)
                feature_entropy += 0 if f == 0 else -f * math.log2(f)
            f = d / len(attribute_types)
            entropy += -f * feature_entropy
        return abs(entropy)

    # perform a partition
    def partitioning(self):
        if decision_tree.flag:
            print('Available partitions')
            for k, v in self.partitions.items():
                print(f'    {k} = {v}')
        f = dict()
        for k, partition in self.partitions.items():
            target_entropy = self.compute_by_target_attribute(partition)
            attribute_entropies = [self.get_attr_entropy(i, partition)
                                   for i in range(self.types - 1)]
            information_gain = [target_entropy - i
                                for i in attribute_entropies]
            f[k] = (len(partition) / self.rows * max(information_gain),
                    information_gain.index(max(information_gain)))
            if decision_tree.flag:
                print(f'\nEntropy({k}) = {target_entropy}')
                for i in range(self.types - 1):
                    print(f'Entropy({k}|A{i}) = {attribute_entropies[i]}\tGain({k}, A{i})'
                          f' = {information_gain[i]}')
                print(f'  F_{k} = {len(partition) / self.rows} * {max(information_gain)} '
                      f'= {f[k][0]}')
        max_f, max_p, max_a = 0, '', 0
        for p, (f, a) in f.items():
            if f > max_f:
                max_f, max_p, max_a = f, p, a
        self.split(max_p, max_a)
        if decision_tree.flag:
            print('New partitions')
            for k, v in self.partitions.items():
                print(f'    {k} = {v}')
            print()

    # splits the partition that has the maximum F-value
    def split(self, partition: str, attribute: int):
        rows = self.partitions[partition]
        del self.partitions[partition]
        unique_vals_attribute = set(self.attribute[i - 1][attribute] for i in rows)
        new_partition_id = self.create_new_id(partition, len(unique_vals_attribute))
        for i, available in enumerate(unique_vals_attribute):
            self.partitions[new_partition_id[i]] = [j for j in rows if self.attribute[j - 1][attribute]
                                                    == available]
        print(f'Partition'
              f' {partition}'
              f' was replaced with partitions '
              f'{", ".join(new_partition_id)}'
              f' using Feature'
              f' {attribute + 1}.')

    # store all the nodes in partitions after splitting them
    def available_nodes(self, fname: str):
        with open(fname, 'w') as file:
            file.writelines([f'{i} {" ".join(map(str, j))}\n' for i, j in self.partitions.items()])

    @classmethod
    def load_from_dataset(cls, fname: str) -> 'decision_tree':
        with open(fname) as f:
            data = f.readlines()
        attribute = []
        j, k = map(int, data[0].split())
        for i in range(1, j + 1):
            row = list(map(int, data[i].split()))
            attribute.append(row)
        return decision_tree(attribute)

    # store a new partition that created from the original data-set
    def set_partition(self):
        self.partitions['A'] = list(range(1, self.rows + 1))

    # create new ID for each partition after the split
    def create_new_id(self, partition: str, n: int) -> List[int]:
        set_obj = set('abcdefgh')
        for k in self.partitions:
            set_obj.discard(k[0])
        set_obj.discard(partition)
        return sorted(set_obj)[:n] if len(set_obj) >= n else [f'{partition}{i}' for i in range(n)]


