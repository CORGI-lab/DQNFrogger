'''
@SumTree
used to store errors with the (s,a,s2.r,t)  tuples
'''

import numpy as np


class SumTree:
    # index of the last place data inserted
    num_of_data = 0

    def __init__(self, size):
        self.size = size
        # create  tree to keep sm of priorities
        self.tree = np.zeros(2 * size - 1)
        # create leafs to keep experience data
        self.data = np.zeros(size, dtype=object)

    '''
    @:param priority : priority of the data tuple 
    @:param data     : data tuple to store in the leaf to use as replay 
                       (state, actionValue, reward, newState, terminalReached)
    '''
    def insert(self, priority, replay_data):

        # calculate tree index to map data
        tree_index = self.num_of_data + self.size - 1

        # add data
        self.data[self.num_of_data] = replay_data

        # update tree with newly added priority
        self.update_tree(tree_index, priority)

        # increase size
        self.num_of_data += 1
        # overwrite data if size max
        if self.num_of_data >= self.size:
            self.num_of_data = 0

    '''
        update tree with new priority 
        @:param tree_index : index of new priority data
        @:param priority   : new data priority 
    '''
    def update_tree(self, tree_index, priority):

        # diff of new priority
        change = priority - self.tree[tree_index]
        # update priority
        self.tree[tree_index] = priority

        # propagate change to root
        while tree_index != 0:
            # cal parent
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    '''
        Get a leaf (experience) given priority 
        @:param priority_value : segment values calculated im memory
        @:return exp           : 
    '''
    def get_exp(self, priority_value):

        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            # downward search, always search for a higher priority node
            else:

                if priority_value <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    priority_value -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.size + 1

        return data_index, leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        # root node has the sum of all priority
        # used to calculate segment size
        return self.tree[0]


