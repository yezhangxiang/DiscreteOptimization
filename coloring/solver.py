#!/usr/bin/python
# -*- coding: utf-8 -*-

import copy
import time
import random
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
    color_num = node_count
    color_constrain = [[1 for col in range(color_num)] for row in range(node_count)]

    tmp_graph = copy.deepcopy(graph)
    tmp_degree = copy.deepcopy(degree)
    solution = greedy(solution, color_constrain, tmp_degree, color_num, tmp_graph)
    color_num = max(solution) + 1
    print(color_num)
    print(solution)

    # best_solution, best_fitness = simulated_annealing(solution, combine_fitness, combine_swap, graph)
    # print(best_fitness)
    # color_num = len(set(solution))

    # color_num_dic = get_color_num(solution)
    # color_num_dic = sorted(color_num_dic.items(), lambda x, y: cmp(x[1], y[1]))
    # color_num_list = [v[0] for v in color_num_dic]
    # print(solution)
    # print(feasibility_fitness(solution))

    tabu_size = 40
    tabu_list = [[-1 for col in range(tabu_size)] for row in range(node_count)]
    tabu_fitness_list = [[-1 for col in range(tabu_size)] for row in range(node_count)]
    print(combine_fitness(solution, graph))
    best_fitness = combine_fitness(solution, graph)
    best_solution = copy.deepcopy(solution)
    for iter in range(10000):
        for node in range(node_count):
            # best_color = solution[node]
            color_size = max(solution) + 1
            neighbor_fitness = [-1] * color_size
            for color in range(color_size):
                solution[node] = color
                new_fitness = combine_fitness(solution, graph)
                neighbor_fitness[color] = new_fitness
                # if new_fitness > best_fitness:
                # best_fitness = new_fitness
                #     best_solution = copy.deepcopy(solution)
                #     best_color = color
            # best_color = neighbor_fitness.index(max(neighbor_fitness))
            select_color, select_fitness = select(tabu_list, tabu_fitness_list, node, neighbor_fitness)
            solution[node] = select_color
            add_tabu(tabu_list, tabu_fitness_list, node, select_color, select_fitness)
            if neighbor_fitness[select_color] > best_fitness:
                best_fitness = neighbor_fitness[select_color]
                best_solution = copy.deepcopy(solution)
                print(best_fitness)
                print(best_solution)
                # print([node, best_color])
                # if solution[node] != color:
                # old_solution = copy.deepcopy(solution)
                # # old_fitness = feasibility_fitness(solution)
                # kemp_chains(node, color, solution, graph)
                # new_fitness = feasibility_fitness(solution)
                # if new_fitness <= old_fitness:
                # solution = copy.copy(old_solution)
                # else:
                # break

        print(solution)
        # print(feasibility_fitness(solution))
        print(combine_fitness(solution, graph))

    # best_solution = solution
    color_num = max(solution) + 1

    # first_node = degree.index(max(degree))
    # color_constrain = [[1 for col in range(color_num)] for row in range(node_count)]
    # solution = [-1] * node_count
    # start = time.clock()
    # (color_num, best_solution) = recursion(color_num, [], first_node, solution, color_constrain, graph, degree, start)

    # prepare the solution in the specified output format
    output_data = str(color_num) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, best_solution))

    return output_data


def coloring_one_node(color_size, node_index, color_index, solution, color_constrain, graph, degree):
    solution[node_index] = color_index
    for i in range(len(solution)):
        if i != solution[node_index] and i < len(color_constrain[node_index]):
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


def recursion(color_num, best_solution, node, solution, color_constrain, graph, degree, start):
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

    # color_candidates = sorted(color_candidates.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    # color_candidates = [v[0] for v in color_candidates]
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
                                                           color_constrain, graph, degree, start)
                    now = time.clock()
                    if now - start > 600 and len(best_solution) > 0:
                        return color_num, best_solution
                else:
                    color_num = len(tmp)
                    # print(color_num)
                    best_solution = copy.copy(solution)
                    return color_num, best_solution
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


def greedy(solution, color_constrain, degree, color_num, graph):
    while -1 in solution:
        node = find_next_node(solution, color_constrain, color_num, degree)
        color = color_constrain[node].index(1)
        coloring_one_node(color_num, node, color, solution, color_constrain, graph, degree)
    return solution


def kemp_chains(node, color, solution, graph):
    old_color_nodes = {node: True}
    new_color_nodes = {}
    old_color = solution[node]
    while not chain_stop(old_color_nodes, new_color_nodes):
        chain(color, solution, graph, old_color_nodes, new_color_nodes)
        chain(old_color, solution, graph, new_color_nodes, old_color_nodes)
    for i in old_color_nodes:
        solution[i] = color
    for i in new_color_nodes:
        solution[i] = old_color


def chain(color, solution, graph, old_color_nodes, new_color_nodes):
    for node in old_color_nodes:
        if old_color_nodes[node]:
            for neighbor in range(len(graph[node])):
                if graph[node][neighbor] == 1 and solution[neighbor] == color and neighbor not in new_color_nodes:
                    new_color_nodes[neighbor] = True
        old_color_nodes[node] = False


def chain_stop(old_color_nodes, new_color_nodes):
    for i in old_color_nodes:
        if old_color_nodes[i]:
            return False
    for i in new_color_nodes:
        if new_color_nodes[i]:
            return False
    return True


def combine_swap(node, color, solution, graph):
    solution[node] = color


def combine_fitness(solution, graph):
    color_num = get_color_num(solution)
    fitness = 0
    for i in color_num:
        bad_edge = 0
        for node in range(len(solution)):
            if solution[node] == i:
                for neighbor in range(len(solution)):
                    if graph[node][neighbor] == 1 and solution[neighbor] == i:
                        bad_edge += 1
        fitness += 2 * bad_edge * color_num[i] - color_num[i] * color_num[i]
    return -fitness


def feasibility_fitness(solution):
    color_num = get_color_num(solution)
    fitness = 0
    for i in color_num:
        fitness += color_num[i] * color_num[i]
    return fitness


def get_color_num(solution):
    color_num = {}
    for i in solution:
        if i in color_num:
            color_num[i] += 1
        else:
            color_num[i] = 1
    return color_num


def simulated_annealing(solution, fitness_fuc, swap_fun, graph):
    max_iter = 1000000
    t = 100
    delta = t / max_iter
    best_solution = copy.deepcopy(solution)
    best_fitness = fitness_fuc(solution, graph)
    print(best_fitness)
    for it in range(max_iter):
        # for node in range(len(solution)):
        # for color in range(max(solution)+1):
        node = random.randrange(0, len(solution))
        color = random.randrange(0, max(solution) + 1)
        if color != solution[node]:
            solution, fitness = metropolis(node, color, solution, fitness_fuc, swap_fun, graph, t)
            if fitness > best_fitness:
                best_fitness = fitness
                print(best_fitness)
                best_solution = copy.deepcopy(solution)
        t -= delta
    return best_solution, best_fitness


def metropolis(node, color, solution, fitness_fuc, swap_fun, graph, t):
    old_fitness = fitness_fuc(solution, graph)
    old_solution = copy.deepcopy(solution)
    swap_fun(node, color, solution, graph)
    new_fitness = fitness_fuc(solution, graph)
    if new_fitness >= old_fitness:
        return solution, new_fitness
    else:
        if random.random() > math.exp(-(old_fitness - new_fitness) / t):
            return old_solution, old_fitness
        else:
            return solution, new_fitness


def add_tabu(tabu_list, tabu_fitness_list, node, color, fitness):
    if -1 in tabu_list[node]:
        new_index = tabu_list[node].index(-1)
        tabu_list[node][new_index] = color
        tabu_fitness_list[node][new_index] = fitness
    else:
        for i in range(len(tabu_list[node])-1):
            tabu_list[node][i] = tabu_list[node][i + 1]
            tabu_fitness_list[node][i] = tabu_fitness_list[node][i]
        tabu_list[node][len(tabu_list[node]) - 1] = color
        tabu_fitness_list[node][len(tabu_list[node]) - 1] = fitness


def select(tabu_list, tabu_fitness_list, node, neighbor_fitness):
    while 1:
        max_fitness = max(neighbor_fitness)
        select_color = neighbor_fitness.index(max_fitness)
        tabu = False
        for i in range(len(tabu_list[node])):
            if tabu_list[node][i] == select_color and tabu_fitness_list[node][i] == max_fitness:
                tabu = True
                break
        if not tabu:
            return select_color, max_fitness
        else:
            neighbor_fitness[select_color] = -10000


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
        print finish - start
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)'

