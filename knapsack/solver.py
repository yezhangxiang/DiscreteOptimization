#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import copy
from collections import namedtuple

Item = namedtuple("Item", ['index', 'value', 'weight', 'performance_ratio'])

sys.setrecursionlimit(1000000)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1]), float(parts[0])/float(parts[1])))

    items = sorted(items, key=lambda item: item.performance_ratio, reverse=True)

    # a dynamic program algorithm for filling the knapsack
    taken = [0] * len(items)  # it takes items in-order until the knapsack is full

    (value, best_taken) = depthFirst(0, item_count, taken, 0, capacity, items, 0, [])

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, best_taken))
    return output_data


def depthFirst(layer, depth, taken, value, room, items, max_value, best_taken):
    """

    :rtype : object
    """
    if layer == depth:
        if value > max_value:
            max_value = value
            best_taken = copy.copy(taken)
        return max_value, best_taken

    estimate = value
    weight_sum = 0
    last_weight = 0
    last_value = 0
    for i in range(layer, depth):
        if weight_sum < room:
            weight_sum += items[i].weight
            last_weight = items[i].weight
            estimate += items[i].value
            last_value = items[i].value
        else:
            break
    if last_weight > 0:
        estimate -= float(weight_sum - room) / last_weight * last_value
    else:
        estimate -= weight_sum

    if estimate <= max_value:
        return max_value, best_taken

    if layer < depth:
        # go left
        taken[items[layer].index] = 1
        value += items[layer].value
        room -= items[layer].weight
        if room >= 0:
            (max_value, best_taken) = depthFirst(layer + 1, depth, taken, value, room, items, max_value,
                                                 best_taken)
        # backtracking
        taken[items[layer].index] = 0
        value -= items[layer].value
        room += items[layer].weight
        # go right
        (max_value, best_taken) = depthFirst(layer + 1, depth, taken, value, room, items, max_value,
                                             best_taken)
    return max_value, best_taken


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)'

