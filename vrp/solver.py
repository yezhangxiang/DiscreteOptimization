#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from gurobipy import *
import matplotlib.pyplot as plt

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])


def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2)


# Callback - use lazy constraints to eliminate sub-tours

def subtourelim(model, where):
    if where == GRB.callback.MIPSOL:
        selected = []
        # make a list of edges selected in the solution
        for i in range(n):
            sol = model.cbGetSolution([model._vars[i, j] for j in range(n)])
            selected += [(i, j) for j in range(n) if sol[j] > 0.5]
        # find the shortest cycle in the selected edge list
        cycles = subtour(selected)
        for cycle in cycles:
            demand_sum = 0
            for c in cycle:
                demand_sum += customers[c].demand
            if 0 not in cycle or demand_sum > vehicle_capacity:
                # add a subtour elimination constraint
                expr = 0
                # for i in range(len(cycle)):
                #     for j in range(i+1, len(cycle)):
                #         expr += model._vars[cycle[i], cycle[j]]
                for i in range(len(cycle)-1):
                    expr += model._vars[cycle[i], cycle[i+1]]
                expr += model._vars[cycle[-1], cycle[0]]
                model.cbLazy(expr <= len(cycle) - 1)


# Euclidean distance between two points

def distance(points, i, j):
    dx = points[i][0] - points[j][0]
    dy = points[i][1] - points[j][1]
    return math.sqrt(dx * dx + dy * dy)


# Given a list of edges, finds the shortest subtour

def subtour(edges):
    visited = [False] * n
    cycles = []
    lengths = []
    selected = [[] for i in range(n)]
    for x, y in edges:
        selected[x].append(y)
    degree0 = len(selected[0])/2
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

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    global vehicle_capacity
    vehicle_capacity = int(parts[2])

    global customers
    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i - 1, int(parts[0]), float(parts[1]), float(parts[2])))

    # the depot is always the first customer in the input
    depot = customers[0]


    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    m = Model()

    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 400)

    # Create variables
    global n
    n = customer_count
    vars = {}
    for i in range(n):
        for j in range(i + 1):
            vars[i, j] = m.addVar(obj=length(customers[i], customers[j]), vtype=GRB.BINARY,
                                  name='e' + str(i) + '_' + str(j))
            vars[j, i] = vars[i, j]
    m.update()


    # Add degree-2 constraint, and forbid loops

    m.addConstr(quicksum(vars[0, j] for j in range(n)) <= 2*vehicle_count)
    m.addConstr(quicksum(vars[0, j] for j in range(n)) >= 2)
    vars[0, 0].ub = 0
    for i in range(1, n):
        m.addConstr(quicksum(vars[i, j] for j in range(n)) == 2)
        vars[i, i].ub = 0
    m.update()


    # Optimize model

    m._vars = vars
    m.params.LazyConstraints = 1
    m.optimize(subtourelim)

    solution = m.getAttr('x', vars)
    selected = [(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5]
    # assert len(subtour(selected)) == n

    # draw(customers, selected)
    # plt.show()
    obj = m.objVal
    # print('')
    # # print('Optimal tour: %s' % str(subtour(selected)))
    # print('Optimal cost: %g' % m.objVal)
    # print('')
    cycles = subtour(selected)
    # print(cycles)
    #
    # vehicle_tours = []
    #
    # remaining_customers = set(customers)
    # remaining_customers.remove(depot)
    #
    # for v in range(0, vehicle_count):
    #     # print "Start Vehicle: ",v
    #     vehicle_tours.append([])
    #     capacity_remaining = vehicle_capacity
    #     while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
    #         used = set()
    #         order = sorted(remaining_customers, key=lambda customer: -customer.demand)
    #         for customer in order:
    #             if capacity_remaining >= customer.demand:
    #                 capacity_remaining -= customer.demand
    #                 vehicle_tours[v].append(customer)
    #                 # print '   add', ci, capacity_remaining
    #                 used.add(customer)
    #         remaining_customers -= used
    #
    # # checks that the number of customers served is correct
    # assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1
    #
    # # calculate the cost of the solution; for each vehicle the length of the route
    # obj = 0
    # for v in range(0, vehicle_count):
    #     vehicle_tour = vehicle_tours[v]
    #     if len(vehicle_tour) > 0:
    #         obj += length(depot, vehicle_tour[0])
    #         for i in range(0, len(vehicle_tour) - 1):
    #             obj += length(vehicle_tour[i], vehicle_tour[i + 1])
    #         obj += length(vehicle_tour[-1], depot)

    # prepare the solution in the specified output format
    outputData = str(obj) + ' ' + str(0) + '\n'
    for v in range(0, len(cycles)):
        outputData += ' '.join(
            # [str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'
            [str(customer) for customer in cycles[v]]) + ' ' + str(depot.index) + '\n'
    for v in range(len(cycles), vehicle_count):
        outputData += str(depot.index) + ' ' + str(depot.index) + '\n'

    return outputData


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

