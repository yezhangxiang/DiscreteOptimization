#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import deque
import numpy as np
import copy
import random
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from collections import namedtuple
from gurobipy import *

Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    node_count = int(lines[0])

    points = []
    for i in range(1, node_count + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # d = [[float('inf') for col in range(node_count)] for row in range(node_count)]
    # for i in range(node_count):
    #     for j in range(node_count):
    #         if i != j:
    #             d[i][j] = length(points[i], points[j])
    #             d[j][i] = d[i][j]

    points = np.array(points)

    # x = points[:, 0]
    # y = points[:, 1]
    # triang = tri.Triangulation(x, y)
    # plt.figure()
    # plt.gca().set_aspect('equal')
    # plt.triplot(points[:, 0], points[:, 1], triang.triangles, 'bo-')
    # plt.title('triplot of Delaunay triangulation')

    # d_dict = [{} for row in range(node_count)]
    # for edge in triang.edges:
    #     d_dict[edge[0]][edge[1]] = length(points[edge[0]], points[edge[1]])
    #     d_dict[edge[1]][edge[0]] = d_dict[edge[0]][edge[1]]

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    global n
    n = node_count
    m = Model()

    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 600)

    # Create variables

    vars = {}
    for i in range(n):
        for j in range(i+1):
            vars[i,j] = m.addVar(obj=distance(points, i, j), vtype=GRB.BINARY,
                                 name='e'+str(i)+'_'+str(j))
            vars[j,i] = vars[i,j]
    m.update()

    # Add degree-2 constraint, and forbid loops

    for i in range(n):
        m.addConstr(quicksum(vars[i,j] for j in range(n)) == 2)
        vars[i,i].ub = 0
    m.update()

    # Optimize model

    m._vars = vars
    m.params.LazyConstraints = 1
    m.optimize(subtourelim)

    solution = m.getAttr('x', vars)
    selected = [(i,j) for i in range(n) for j in range(n) if solution[i,j] > 0.5]
    assert len(subtour(selected)) == n

    # print('')
    # print('Optimal tour: %s' % str(subtour(selected)))
    # print('Optimal cost: %g' % m.objVal)
    # print('')

    obj = m.objVal
    solution = subtour(selected)

    # solution = greed(d_dict)
    # solution = opt_2(solution, points)
    # solution.reverse()
    # draw(points, solution)

    # index = solution.index(3)
    # solution = opt_k(solution, index, d)
    # for i in range(node_count):
    # index = solution.index(i)
    # solution = opt_k(solution, index, d, 4)
    # draw(points, solution)

    # solution = range(node_count)
    # print(solution)
    # solution = tabu_search(solution, d, triang.edges, points)
    # solution = constraint(d, d_dict, solution)
    # solution = opt_2_5(solution, d)
    # solution, obj = insert_node(points, d_dict)

    # obj = length_tour(solution, d)
    # obj = length_tour2(solution, points, d_dict)
    # draw(points, solution)

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def opt_2_5(solution, points, d_dict):
    round_end = False
    node_count = len(solution)
    current_obj = length_tour2(solution, points, d_dict)
    while not round_end:
        has_improvement = False
        for node in range(node_count):
            node_i = solution.index(node)
            node_before = solution[(node_i - 1) % node_count]
            node_after = solution[(node_i + 1) % node_count]
            best_d = d(points, d_dict, node, node_before) + d(points, d_dict, node, node_after) - \
                     d(points, d_dict, node_before, node_after)
            best_i = node_i
            init_obj = current_obj - best_d
            del solution[node_i]
            for insert_i in range(1, len(solution) + 1):
                node_before = solution[insert_i - 1]
                node_after = solution[insert_i % len(solution)]
                if node_after not in d_dict[node] and node_before not in d_dict[node]:
                    continue
                current_d = d(points, d_dict, node_before, node) + d(points, d_dict, node_after, node) - \
                            d(points, d_dict, node_before, node_after)
                if current_d < best_d:
                    best_d = current_d
                    best_i = insert_i
            solution.insert(best_i, node)
            current_obj = init_obj + best_d
        if not has_improvement:
            round_end = True
    return solution, current_obj


def insert_node(points, d_dict):
    max_gen = 1
    best_obj = float('inf')
    best_solution = []
    node_count = len(points)
    for i in range(max_gen):
        start = time.clock()
        node_index = range(node_count)
        random.shuffle(node_index)
        solution = node_index[:3]
        for node_i in range(3, node_count):
            local_obj = float('inf')
            local_i = 0
            node = node_index[node_i]
            for insert_i in range(1, len(solution) + 1):
                node_before = solution[insert_i - 1]
                node_after = solution[insert_i % len(solution)]
                current_obj = d(points, d_dict, node_before, node) + \
                              d(points, d_dict, node_after, node) - \
                              d(points, d_dict, node_before, node_after)
                if current_obj < local_obj:
                    local_obj = current_obj
                    local_i = insert_i
            solution.insert(local_i, node)
        solution, obj = opt_2_5(solution, points, d_dict)
        # obj = length_tour(solution, d)
        if obj < best_obj:
            best_obj = obj
            best_solution = copy.deepcopy(solution)
        # print(time.clock() - start)
    return best_solution, best_obj


def constraint(d, d_dict, best_solution):
    used = [False] * len(d)
    used[0] = True
    solution = [0]
    obj = 0
    # best_solution = greed(d)
    # best_solution = opt_2(solution, points)
    min_obj = length_tour(best_solution, d)
    print(min_obj)
    min_obj, best_solution = depth_first(d, d_dict, used, solution, obj, min_obj, best_solution)
    print(min_obj)
    return best_solution


def depth_first(d, d_dict, used, solution, obj, min_obj, best_solution):
    if len(solution) == len(used):
        obj += d[solution[0]][solution[-1]]
        if obj < min_obj:
            min_obj = obj
            best_solution = copy.copy(solution)
        return min_obj, best_solution

    estimate = obj
    min_estimate = float('inf')
    for i in range(len(used)):
        if not used[i]:
            d_pair = sorted(d_dict[i].items(), lambda x, y: cmp(x[1], y[1]))
            # estimate += min(d[i])
            i_estimate = (d_pair[0][1] + d_pair[1][1]) / 2
            if d_pair[0][1] < min_estimate:
                min_estimate = d_pair[0][1]
            estimate += i_estimate
    estimate += min_estimate

    if estimate >= min_obj:
        return min_obj, best_solution

    candidate = sorted(d_dict[solution[-1]].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    for next_pair in candidate:
        next_node = next_pair[0]
        if not used[next_node]:
            solution.append(next_node)
            obj += d[solution[-2]][solution[-1]]
            used[next_node] = True
            (min_obj, best_solution) = depth_first(d, d_dict, used, solution, obj, min_obj, best_solution)
            # backtracking
            used[next_node] = False
            obj -= d[solution[-2]][solution[-1]]
            solution.pop()

    return min_obj, best_solution


def tabu_search(solution, d, edges, points):
    best_solution = copy.deepcopy(solution)
    best_obj = length_tour(solution, d)
    t = 0
    max_gen = 1000
    tabu_size = 40
    tabu_list = deque()

    print(best_obj)
    while t < max_gen:
        nn = 0
        local_obj = float('inf')
        for edge in edges:
            foo = solution.index(edge[0])
            bar = solution.index(edge[1])
            tmp_solution = swap(solution, foo, bar)
            # draw(points, tmp_solution)
            plt.show()
            if tmp_solution not in tabu_list:
                tmp_obj = length_tour(tmp_solution, d)
                # tmp_obj = length_tour(tmp_solution, d)
                # if tmp_obj not in tabu_list:
                if tmp_obj < local_obj:
                    local_obj = tmp_obj
                    local_solution = copy.deepcopy(tmp_solution)
                nn += 1
        if local_obj < best_obj:
            best_obj = local_obj
            best_solution = copy.deepcopy(local_solution)
        solution = local_solution

        if len(tabu_list) >= tabu_size:
            tabu_list.popleft()
        tabu_list.append(local_solution)
        # tabu_list.append(local_obj)
        t += 1
        print(best_obj)
        draw(points, best_solution)
    return best_solution


def opt_k(solution, index, d, k):
    best_solution = copy.deepcopy(solution)
    best_obj = length_tour(solution, d)
    tabu = []
    best_obj, best_solution = opt_k_recursion(solution, d, best_obj, best_solution, index, tabu, k)
    return best_solution


def opt_k_recursion(solution, d, best_obj, best_solution, index, tabu, k):
    t1 = solution[index]
    t2 = solution[(index + 1) % len(solution)]
    t4 = solution[(index + 2) % len(solution)]
    if (t2, t4) in tabu or k <= 0:
        return best_obj, best_solution
    tabu.append((t2, t4))
    tmp_d = copy.deepcopy(d[t2])
    old_d = tmp_d[t1]
    tmp_d[t1] = float('inf')
    tmp_d[t4] = float('inf')
    candidate = [i for i in range(len(tmp_d)) if tmp_d[i] < old_d]
    for t3 in candidate:
        next_index = solution.index(t3)
        solution = swap(solution, index, next_index)
        obj = length_tour(solution, d)
        # print('t2: ' + str(t2) + ' t3: ' + str(t3) + ' t4: ' + str(t4) + ' obj: ' + str(obj))
        if obj < best_obj:
            best_obj = obj
            best_solution = copy.deepcopy(solution)
        opt_k_recursion(solution, d, best_obj, best_solution, index, tabu, k - 1)
        solution = swap(solution, index, next_index)
    return best_obj, best_solution


def opt_2(solution, points):
    node_count = len(solution)
    for index in range(0, node_count - 1):
        no_segment_intersect = False
        while not no_segment_intersect:
            segment_intersect = False
            for i in range(2, node_count - 1):
                if intersect([points[solution[index]], points[solution[index + 1]]],
                             [points[solution[(index + i) % node_count]],
                              points[solution[(index + i + 1) % node_count]]]):
                    solution = swap(solution, index, index + i)
                    segment_intersect = True
                    break
            if not segment_intersect:
                no_segment_intersect = True
    return solution


def swap(solution, begin, end):
    begin += 1
    end += 1
    if end < begin:
        tmp = end
        end = begin
        begin = tmp
    tmp = solution[begin:end]
    tmp.reverse()
    return solution[:begin] + tmp + solution[end:]


def swap_point(solution, foo, bar):
    tmp = solution[foo]
    solution[foo] = solution[bar]
    solution[bar] = tmp


def multiply(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])


def intersect(segment1, segment2):
    return (multiply(segment1[0], segment2[1], segment2[0]) *
            multiply(segment2[1], segment1[1], segment2[0]) >= 0.1) and (
        multiply(segment2[0], segment1[1], segment1[0]) *
        multiply(segment1[1], segment2[1],
                 segment1[0]) >= 0.1)


def d(points, d_dict, point1, point2):
    if point2 not in d_dict[point1]:
        return length(points[point1], points[point2])
    return d_dict[point1][point2]


def length_tour2(solution, points, d_dict):
    obj = d(points, d_dict, solution[-1], solution[0])
    for index in range(0, len(solution) - 1):
        obj += d(points, d_dict, solution[index], solution[index + 1])
    return obj


def length_tour(solution, d):
    # calculate the length of the tour
    obj = d[solution[-1]][solution[0]]
    for index in range(0, len(solution) - 1):
        obj += d[solution[index]][solution[index + 1]]
    return obj


def greed(d_dict):
    node_count = len(d_dict)
    used = [False] * node_count
    solution = [-1] * node_count
    used[0] = True
    solution[0] = 0
    for index in range(0, node_count - 1):
        tmp_d = sorted(d_dict[solution[index]].items(), lambda x, y: cmp(x[1], y[1]))
        for item in tmp_d:
            next_node = item[0]
            if not used[next_node]:
                solution[index + 1] = next_node
                used[next_node] = True
                break
        if solution[index+1] == -1:
            next_node = used.index(False)
            solution[index+1] = next_node
            used[next_node] = True
    return solution


def draw(points, solution):
    plt.figure()
    node_count = len(solution)
    x = [v[0] for v in points]
    y = [v[1] for v in points]
    plt.plot(x, y, '.')
    ax = plt.axes()
    for index in range(0, node_count):
        ax.annotate(str(solution[index]),
                    xy=(points[solution[(index + 1) % node_count]][0], points[solution[(index + 1) % node_count]][1]),
                    xytext=(points[solution[index]][0], points[solution[index]][1]))
        ax.annotate("",
                    xy=(points[solution[(index + 1) % node_count]][0], points[solution[(index + 1) % node_count]][1]),
                    xytext=(points[solution[index]][0], points[solution[index]][1]), arrowprops=dict(arrowstyle="->"))
    ax.axis('equal')


# Callback - use lazy constraints to eliminate sub-tours

def subtourelim(model, where):
    if where == GRB.callback.MIPSOL:
        selected = []
        # make a list of edges selected in the solution
        for i in range(n):
            sol = model.cbGetSolution([model._vars[i,j] for j in range(n)])
            selected += [(i,j) for j in range(n) if sol[j] > 0.5]
        # find the shortest cycle in the selected edge list
        tour = subtour(selected)
        if len(tour) < n:
            # add a subtour elimination constraint
            expr = 0
            for i in range(len(tour)):
                for j in range(i+1, len(tour)):
                    expr += model._vars[tour[i], tour[j]]
            model.cbLazy(expr <= len(tour)-1)


# Euclidean distance between two points

def distance(points, i, j):
    dx = points[i][0] - points[j][0]
    dy = points[i][1] - points[j][1]
    return math.sqrt(dx*dx + dy*dy)


# Given a list of edges, finds the shortest subtour

def subtour(edges):
    visited = [False]*n
    cycles = []
    lengths = []
    selected = [[] for i in range(n)]
    for x,y in edges:
        selected[x].append(y)
    while True:
        current = visited.index(False)
        thiscycle = [current]
        while True:
            visited[current] = True
            neighbors = [x for x in selected[current] if not visited[x]]
            if len(neighbors) == 0:
                break
            current = neighbors[0]
            thiscycle.append(current)
        cycles.append(thiscycle)
        lengths.append(len(thiscycle))
        if sum(lengths) == n:
            break
    return cycles[lengths.index(min(lengths))]
import sys
import time

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        start = time.clock()
        print solve_it(input_data)
        finish = time.clock()
        # print(finish - start)
        plt.show()
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)'

