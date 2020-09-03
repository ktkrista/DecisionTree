import argparse

# build a command line interface
parser = argparse.ArgumentParser()
parser.add_argument('-f', dest='flag', action='store_true', default=False)
parser.add_argument('-epsilon', default=0)
parser.add_argument('data_set')
parser.add_argument('partition_input')
parser.add_argument('partition_output', action='store')
args = parser.parse_args()

from partition import decision_tree

# read in data, perform partitions
# and output the results to the file
decision_tree.epsilon = args.epsilon
decision_tree.flag = args.flag
decision_tree = decision_tree.load_from_dataset(args.data_set)
try:
    decision_tree.read_partition_file(args.partition_input)
# if partition file cannot be found,
except OSError as error:
    print(f"'{args.partition_input}'"
          f" cannot be opened")
    # partition the data set file instead
    print('Generate new partition from the data set: ')
    decision_tree.set_partition()
decision_tree.partitioning()
decision_tree.available_nodes(args.partition_output)
