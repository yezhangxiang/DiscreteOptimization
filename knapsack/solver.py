#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # a dynamic program algorithm for filling the knapsack
    taken = [0]*len(items)    # it takes items in-order until the knapsack is full
    pd = [[0 for col in range(item_count+1)] for row in range(capacity+1)]

    for item in items:
        for i in range(capacity+1):
            if item.weight <= i:
                pd[i][item.index+1] = max(pd[i][item.index], item.value + pd[i-item.weight][item.index])
            else:
                pd[i][item.index+1] = pd[i][item.index]

    value = pd[capacity][item_count]

    tmp_capacity = capacity
    for i in range(item_count, 1, -1):
        if pd[tmp_capacity][i] != pd[tmp_capacity][i-1]:
            taken[i-1] = 1
            tmp_capacity = tmp_capacity - items[i-1].weight

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


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

