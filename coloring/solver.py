#!/usr/bin/python
# -*- coding: utf-8 -*-

import copy
import time
import math


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    graph = [[0 for col in range(node_count)] for row in range(node_count)]
    color_constrain = [[1 for col in range(node_count)] for row in range(node_count)]
    degree = [0] * node_count
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
        graph[int(parts[0])][int(parts[1])] = 1
        graph[int(parts[1])][int(parts[0])] = 1
        degree[int(parts[0])] += 1
        degree[int(parts[1])] += 1

    solution = [-1] * node_count

    # 给度最大的上0色
    first_node = degree.index(max(degree))

    color_num = node_count
    (color_num, best_solution) = recursion(color_num, [], first_node, solution, color_constrain, graph, degree)

    # prepare the solution in the specified output format
    output_data = str(color_num) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, best_solution))

    return output_data


def coloring_one_node(color_size, node_index, color_index, solution, color_constrain, graph, degree):
    solution[node_index] = color_index
    for i in range(len(solution)):
        if i != solution[node_index]:
            color_constrain[node_index][i] = 0
        if graph[node_index][i] == 1:
            color_constrain[i][solution[node_index]] = 0
            graph[node_index][i] = 0
            graph[i][node_index] = 0
            degree[i] -= 1
            degree[node_index] -= 1
    if is_valid(color_constrain, color_size) is True:
        for i in range(len(solution)):
            if solution[i] == -1 and sum(color_constrain[i][:color_size]) == 1:
                if coloring_one_node(color_size, i, color_constrain[i][:color_size].index(1), solution, color_constrain,
                                     graph, degree) is False:
                    return False
            if solution[i] == -1 and degree[i] == 0:
                if sum(color_constrain[i][:color_size]) == 0:
                    print("impossible again")
                    return False
                if coloring_one_node(color_size, i, color_constrain[i][:color_size].index(1), solution, color_constrain,
                                     graph, degree) is False:
                    return False
    else:
        return False
    return True


def find_next_node(solution, color_constrain, color_num, degree):
    next_node = -1
    min_domain = len(solution) + 1
    max_degree = -1
    for i in range(len(solution)):
        if solution[i] == -1:
            domain = sum(color_constrain[i][:color_num])
            if domain < min_domain:
                min_domain = domain
                max_degree = degree[i]
                next_node = i
            if domain == min_domain and degree[i] > max_degree:
                max_degree = degree[i]
                next_node = i
    return next_node


def recursion(color_num, best_solution, node, solution, color_constrain, graph, degree):
    if sum(color_constrain[node][:color_num]) == 0:
        print("end")
    color_i = int(0)
    color_candidates = {}
    while color_i < min([max(solution) + 2, color_num]):
        if color_constrain[node][color_i] == 1:
            color_neighbor_num = 0
            for node_i in range(len(solution)):
                if graph[node][node_i] == 1 and color_i in color_constrain[node_i]:
                    color_neighbor_num += 1
            color_candidates[color_i] = color_neighbor_num
        color_i += 1

    sorted(color_candidates.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    color_candidates = list(color_candidates.keys())

    for i in color_candidates:
        tmp_solution = copy.deepcopy(solution)
        tmp_color_constrain = copy.deepcopy(color_constrain)
        tmp_graph = copy.deepcopy(graph)
        tmp_degree = copy.deepcopy(degree)
        if coloring_one_node(color_num, node, i, solution, color_constrain, graph, degree) is True:
            tmp = list(set(solution))
            if -1 in tmp:
                tmp.remove(-1)
            if len(tmp) < color_num:
                if -1 in solution:
                    next_node = find_next_node(solution, color_constrain, color_num, degree)
                    (color_num, best_solution) = recursion(color_num, best_solution, next_node, solution,
                                                           color_constrain, graph, degree)
                else:
                    color_num = len(tmp)
                    best_solution = copy.copy(solution)
        solution = copy.copy(tmp_solution)
        color_constrain = copy.copy(tmp_color_constrain)
        graph = copy.copy(tmp_graph)
        degree = copy.copy(tmp_degree)
    return color_num, best_solution


def is_valid(color_constrain, color_num):
    for con in color_constrain:
        if sum(con[:color_num]) == 0:
            return False
    return True


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        start = time.clock()
        print solve_it(input_data)
        finish = time.clock()
        #print finish - start
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)'

