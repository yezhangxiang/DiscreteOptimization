#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])


def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2)


# Callback - use lazy constraints to eliminate sub-tours

def subtourelim(model, where):
    if where == GRB.callback.MIPSOL:
        selected = []
        for v in range(vehicle_count):
            # make a list of edges selected in the solution
            for i in range(customer_count):
                sol = model.cbGetSolution([model._vars[v, i, j] for j in range(customer_count)])
                selected += [(i, j) for j in range(customer_count) if sol[j] > 0.5]
            # find the shortest cycle in the selected edge list
            cycles = subtour(selected)
            if len(cycles) > 1:
                for cycle in cycles:
                    if 0 not in cycle:
                        for v2 in range(vehicle_count):
                            expr = 0
                            for i in range(len(cycle) - 1):
                                expr += model._vars[v2, cycle[i], cycle[i + 1]]
                            expr += model._vars[v2, cycle[-1], cycle[0]]
                            model.cbLazy(expr <= len(cycle) - 1)


# Euclidean distance between two points

def distance(points, i, j):
    dx = points[i][0] - points[j][0]
    dy = points[i][1] - points[j][1]
    return math.sqrt(dx * dx + dy * dy)


# Given a list of edges, finds the shortest subtour

def subtour(edges):
    visited = [True] * customer_count
    cycles = []
    lengths = []
    selected = [[] for i in range(customer_count)]
    for x, y in edges:
        selected[x].append(y)
        visited[x] = False
    degree0 = len(selected[0]) / 2
    while True:
        if False not in visited:
            break
        if degree0 > 0:
            visited[0] = False
            degree0 -= 1
        current = visited.index(False)
        thiscycle = [current]
        while True:
            visited[current] = True
            neighbors = [x for x in selected[current] if not visited[x]]
            if len(neighbors) == 0:
                break
            if neighbors[0] == 0 and len(neighbors) > 1:
                current = neighbors[1]
            else:
                current = neighbors[0]
            thiscycle.append(current)
        cycles.append(thiscycle)
        lengths.append(len(thiscycle))
    return cycles


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    global customer_count
    global vehicle_count
    global vehicle_capacity
    global customers
    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i - 1, int(parts[0]), float(parts[1]), float(parts[2])))

    # the depot is always the first customer in the input
    depot = customers[0]

    vehicle_tours = two_phase(customers)

    # vehicle_tours = clark_wright(customers)
    clark_wright_count = len(vehicle_tours)
    if clark_wright_count < vehicle_count:
        for i in range(clark_wright_count, vehicle_count):
            vehicle_tours.append([0, 0])
    # print(len(vehicle_tours))

    obj = 0
    selects = []
    for v in vehicle_tours:
        for i in range(len(v) - 1):
            obj += length(customers[v[i]], customers[v[i + 1]])
            selects.append([v[i], v[i+1]])
    draw(customers, selects)
    plt.show()


    # prepare the solution in the specified output format
    outputData = str(obj) + ' ' + str(0) + '\n'
    for v in range(0, len(vehicle_tours)):
        outputData += ' '.join(
            # [str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'
            [str(customer) for customer in vehicle_tours[v]]) + ' ' + '\n'
    for v in range(len(vehicle_tours), vehicle_count):
        outputData += str(depot.index) + ' ' + str(depot.index) + '\n'

    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    # m = Model()
    #
    # # m.setParam("OutputFlag", 0)
    # # m.setParam("TimeLimit", 1000)
    #
    # # Create variables
    # vars = {}
    # veh_cus = {}
    # for v in range(vehicle_count):
    # for i in range(customer_count):
    #         for j in range(i + 1):
    #             vars[v, i, j] = m.addVar(obj=length(customers[i], customers[j]), vtype=GRB.BINARY,
    #                                   name='e' + str(v) + str(i) + '_' + str(j))
    #             vars[v, j, i] = vars[v, i, j]
    #     for c in range(customer_count):
    #         veh_cus[v, c] = m.addVar(vtype=GRB.BINARY, name='v'+str(v)+'_'+str(c))
    # m.update()
    #
    #
    # # Add degree-2 constraint, and forbid loops
    #
    # for v in range(vehicle_count):
    #     m.addConstr(quicksum(veh_cus[v, c] * customers[c].demand for c in range(customer_count)) <= vehicle_capacity)
    #     for i in range(customer_count):
    #         m.addConstr(quicksum(vars[v, i, j] for j in range(customer_count)) == 2*veh_cus[v, i])
    #         vars[v, i, i].ub = 0
    #
    # for c in range(1, customer_count):
    #     m.addConstr(quicksum(veh_cus[v, c] for v in range(vehicle_count)) == 1)
    # for v in range(vehicle_count):
    #     m.addConstr(quicksum(veh_cus[v, c] for c in range(1, customer_count)) <= veh_cus[v, 0] * customer_count)
    # m.update()
    #
    #
    # # Optimize model
    #
    # m._vars = vars
    # m.params.LazyConstraints = 1
    # m.optimize(subtourelim)
    #
    # solution = m.getAttr('x', vars)
    # cycles = []
    # all_edge = []
    # for v in range(vehicle_count):
    #     selected = [(i, j) for i in range(customer_count) for j in range(customer_count) if solution[v, i, j] > 0.5]
    #     this_cycle = subtour(selected)
    #     cycles += this_cycle
    #     all_edge += selected
    # # assert len(subtour(selected)) == n
    #
    # # draw(customers, all_edge)
    # # plt.show()
    # obj = m.objVal
    #
    # # prepare the solution in the specified output format
    # outputData = str(obj) + ' ' + str(0) + '\n'
    # for v in range(0, len(cycles)):
    #     outputData += ' '.join(
    #         # [str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'
    #         [str(customer) for customer in cycles[v]]) + ' ' + str(depot.index) + '\n'
    # for v in range(len(cycles), vehicle_count):
    #     outputData += str(depot.index) + ' ' + str(depot.index) + '\n'

    return outputData


def two_phase(customers):
    # phase 1: Cluster First, Sweep algorithm
    azimuth = {}
    depot = customers[0]
    seed = customers[1]
    v_seed = [seed.x - depot.x, seed.y - depot.y]
    for i in range(1, len(customers)):
        customer = customers[i]
        v = [customer.x-depot.x, customer.y-depot.y]
        cos_theta = (v[0]*v_seed[0]+v[1]*v_seed[1])/(math.sqrt((v[0]**2+v[1]**2)*(v_seed[0]**2 + v_seed[1]**2)))
        theta = math.acos(cos_theta)
        if multiply(depot, seed, customer) < 0:
            theta = -theta
        azimuth[customer] = theta

    azimuth = sorted(azimuth.items(), lambda x, y: cmp(x[1], y[1]))
    clusters = []
    cluster_demand = 0
    cluster = []
    for node in azimuth:
        cluster_demand += node[0].demand
        if cluster_demand <= vehicle_capacity:
            cluster.append(node[0])
        else:
            cluster.append(depot)
            clusters.append(cluster)
            cluster = [node[0]]
            cluster_demand = node[0].demand
    cluster.append(depot)
    clusters.append(cluster)

    vehicle_tours = []
    for cluster in clusters:
        points = []
        for i in range(len(cluster)):
            points.append([cluster[i].x , cluster[i].y])
        points = np.array(points)
        tour = tsp(points)
        tour_map = []
        for i in tour:
            tour_map.append(cluster[i].index)
        vehicle_tours.append(tour_map)
    for i in range(len(vehicle_tours)):
        tour = vehicle_tours[i]
        index0 = tour.index(0)
        tmp_tour = tour[index0:] + tour[:index0] + [0]
        vehicle_tours[i] = tmp_tour
    return vehicle_tours


def multiply(p0, p1, p2):
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)


def tsp(points):
    node_count = len(points)
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

    # obj = m.objVal
    solution = subtour(selected)
    return solution


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


def clark_wright(customers):
    customer_count = len(customers)
    table = [[0 for col in range(customer_count)] for row in range(customer_count)]
    # Step 1. Saving computation
    for i in range(customer_count):
        for j in range(i):
            table[i][j] = length(customers[i], customers[j])

    saving_list = {}
    for i in range(1, customer_count):
        for j in range(i + 1, customer_count):
            saving_list[(i, j)] = table[i][0] + table[j][0] - table[j][i]

    saving_list = sorted(saving_list.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

    vehicle_tours = []
    for i in range(1, customer_count):
        vehicle_tours.append([0, i, 0])
    for save in saving_list:
        points_pair = save[0]
        start0 = []
        start1 = []
        end0 = []
        end1 = []
        for tour in vehicle_tours:
            if tour[1] == points_pair[0]:
                start0 = tour
            if tour[1] == points_pair[1]:
                start1 = tour
            if tour[-2] == points_pair[0]:
                end0 = tour
            if tour[-2] == points_pair[1]:
                end1 = tour
        if len(start0) and len(end1) and len(list(set(start0[1:-2]).intersection(set(end1[1:-2])))) == 0:
            merged = start0[:-1] + end1[1:]
            demand_sum = 0
            for customer in set(merged):
                demand_sum += customers[customer].demand
            if demand_sum <= vehicle_capacity:
                vehicle_tours.remove(start0)
                vehicle_tours.remove(end1)
                vehicle_tours.append(merged)
        elif len(start1) and len(end0) and len(list(set(start1[1:-2]).intersection(set(end0[1:-2])))) == 0:
            merged = end0[:-1] + start1[1:]
            demand_sum = 0
            for customer in set(merged):
                demand_sum += customers[customer].demand
            if demand_sum <= vehicle_capacity:
                vehicle_tours.remove(start1)
                vehicle_tours.remove(end0)
                vehicle_tours.append(merged)
    return vehicle_tours


def draw(points, selected):
    plt.figure()
    x = [v.x for v in points]
    y = [v.y for v in points]
    plt.plot(x, y, '.')
    ax = plt.axes()
    for p in points:
        ax.annotate(str(p.index),
                    xy=(p.x, p.y),
                    xytext=(p.x, p.y))

    for line in selected:
        ax.annotate("",
                    xy=(points[line[0]].x, points[line[0]].y),
                    xytext=(points[line[1]].x, points[line[1]].y), arrowprops=dict(arrowstyle="-"))
    ax.axis('equal')


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print 'Solving:', file_location
        print solve_it(input_data)
    else:

        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)'

