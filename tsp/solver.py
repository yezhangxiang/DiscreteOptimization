#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import copy
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

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = greed(d)
    solution = opt_2(solution, points)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, node_count - 1):
        obj += d[solution[index]][solution[index + 1]]

    # plt.figure()
    # x = [v[0] for v in points]
    # y = [v[1] for v in points]
    # plt.plot(x, y, '.')
    # ax = plt.axes()
    # for index in range(0, node_count):
    #     ax.annotate(str(solution[index]),
    #                 xy=(points[solution[(index + 1) % node_count]][0], points[solution[(index + 1) % node_count]][1]),
    #                 xytext=(points[solution[index]][0], points[solution[index]][1]))
    #     ax.annotate("", xy=(points[solution[(index + 1) % node_count]][0], points[solution[(index + 1) % node_count]][1]),
    #                 xytext=(points[solution[index]][0], points[solution[index]][1]), arrowprops=dict(arrowstyle="->"))
    #
    # ax.axis('equal')
    # plt.show()


    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


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


def opt_2(solution, points):
    node_count = len(solution)
    for index in range(0, node_count - 1):
        for i in range(2, node_count - 1):
            if intersect([points[solution[index]], points[solution[index + 1]]],
                         [points[solution[(index + i) % node_count]], points[solution[(index + i + 1) % node_count]]]):
                # print([solution[index], solution[index + 1],
                # solution[(index + i) % node_count], solution[(index + i + 1) % node_count]])
                # print(solution)
                solution = swap(solution, index, i)
    return solution


def swap(solution, index, i):
    begin = index + 1
    end = (index + i + 1) % len(solution)
    if end < index:
        begin = end
        end = index + 1
    tmp = solution[begin:end]
    tmp.reverse()
    return solution[:begin] + tmp + solution[end:]


def multiply(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])


def intersect(segment1, segment2):
    return (multiply(segment1[0], segment2[1], segment2[0]) *
            multiply(segment2[1], segment1[1], segment2[0]) >= 0) and (multiply(segment2[0], segment1[1], segment1[0]) *
                                                                       multiply(segment1[1], segment2[1],
                                                                                segment1[0]) >= 0)


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)'

