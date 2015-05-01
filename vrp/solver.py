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
                            for i in range(len(cycle)-1):
                                expr += model._vars[v2, cycle[i], cycle[i+1]]
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


    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    m = Model()

    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 400)

    # Create variables
    vars = {}
    veh_cus = {}
    for v in range(vehicle_count):
        for i in range(customer_count):
            for j in range(i + 1):
                vars[v, i, j] = m.addVar(obj=length(customers[i], customers[j]), vtype=GRB.BINARY,
                                      name='e' + str(v) + str(i) + '_' + str(j))
                vars[v, j, i] = vars[v, i, j]
        for c in range(customer_count):
            veh_cus[v, c] = m.addVar(vtype=GRB.BINARY, name='v'+str(v)+'_'+str(c))
    m.update()


    # Add degree-2 constraint, and forbid loops

    for v in range(vehicle_count):
        m.addConstr(quicksum(veh_cus[v, c] * customers[c].demand for c in range(customer_count)) <= vehicle_capacity)
        for i in range(customer_count):
            m.addConstr(quicksum(vars[v, i, j] for j in range(customer_count)) == 2*veh_cus[v, i])
            vars[v, i, i].ub = 0

    for c in range(1, customer_count):
        m.addConstr(quicksum(veh_cus[v, c] for v in range(vehicle_count)) == 1)
    for v in range(vehicle_count):
        m.addConstr(quicksum(veh_cus[v, c] for c in range(1, customer_count)) <= veh_cus[v, 0] * customer_count)
    m.update()


    # Optimize model

    m._vars = vars
    m.params.LazyConstraints = 1
    m.optimize(subtourelim)

    solution = m.getAttr('x', vars)
    cycles = []
    all_edge = []
    for v in range(vehicle_count):
        selected = [(i, j) for i in range(customer_count) for j in range(customer_count) if solution[v, i, j] > 0.5]
        this_cycle = subtour(selected)
        cycles += this_cycle
        all_edge += selected
    # assert len(subtour(selected)) == n

    # draw(customers, all_edge)
    # plt.show()
    obj = m.objVal

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

