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

Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


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

    d = [[float('inf') for col in range(node_count)] for row in range(node_count)]
    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                d[i][j] = length(points[i], points[j])
                d[j][i] = d[i][j]

    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    triang = tri.Triangulation(x, y)
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(points[:, 0], points[:, 1], triang.triangles, 'bo-')
    plt.title('triplot of Delaunay triangulation')

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    # solution = greed(d)
    # solution = opt_2(solution, points)
    # solution.reverse()
    # draw(points, solution)

    # index = solution.index(3)
    # solution = opt_k(solution, index, d)
    # for i in range(node_count):
    # index = solution.index(i)
    # solution = opt_k(solution, index, d, 4)
    # draw(points, solution)

    solution = range(node_count)
    solution = tabu_search(solution, d)
    draw(points, solution)
    plt.show()

    obj = length_tour(solution, d)

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


# def constraint(points):


def tabu_search(solution, d):
    best_solution = copy.deepcopy(solution)
    best_obj = length_tour(solution, d)
    t = 0
    max_gen = 1000
    N = 200
    tabu_size = 20
    tabu_list = deque()

    while t < max_gen:
        nn = 0
        local_obj = float('inf')
        while nn < N:
            foo = random.randint(0, len(solution) - 1)
            bar = random.randint(0, len(solution) - 1)
            tmp_solution = swap(solution, foo, bar)
            if tmp_solution not in tabu_list:
                tmp_obj = length_tour(tmp_solution, d)
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
        t += 1
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
    if end > begin:
        tmp = solution[begin:end]
        tmp.reverse()
        return solution[:begin] + tmp + solution[end:]
    else:
        tmp_end = end + len(solution)
        double_solution = solution + solution
        tmp = double_solution[begin: tmp_end]
        tmp.reverse()
        return tmp[-end:] + solution[end:begin] + tmp[:-end]


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


def length_tour(solution, d):
    # calculate the length of the tour
    obj = d[solution[-1]][solution[0]]
    for index in range(0, len(solution) - 1):
        obj += d[solution[index]][solution[index + 1]]
    return obj


def greed(d):
    node_count = len(d)
    used = [False] * node_count
    used[0] = True
    solution = range(0, node_count)
    for index in range(0, node_count - 1):
        tmp_d = copy.deepcopy(d[solution[index]])
        while 1:
            next_node = tmp_d.index(min(tmp_d))
            if not used[next_node]:
                solution[index + 1] = next_node
                used[next_node] = True
                break
            else:
                tmp_d[next_node] = float('inf')
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
        print(finish - start)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)'

