#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
from gurobipy import *

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # build a trivial solution
    # pack the facilities one by one until all the customers are served


    # Warehouse demand in thousands of units
    # demand = [15, 18, 14, 20]
    demand = [v.demand for v in customers]

    # Plant capacity in thousands of units
    # capacity = [20, 22, 17, 19, 18]
    capacity = [v.capacity for v in facilities]

    # Fixed costs for each plant
    # fixedCosts = [12000, 15000, 17000, 13000, 16000]
    fixedCosts = [v.setup_cost for v in facilities]

    # Transportation costs per thousand units
    # transCosts = [[4000, 2000, 3000, 2500, 4500],
    #               [2500, 2600, 3400, 3000, 4000],
    #               [1200, 1800, 2600, 4100, 3000],
    #               [2200, 2600, 3100, 3700, 3200]]
    transCosts = [[float('inf') for col in range(facility_count)] for row in range(customer_count)]
    for customer_i in range(customer_count):
        for facility_i in range(facility_count):
            transCosts[customer_i][facility_i] = length(customers[customer_i].location, facilities[facility_i].location)

    # Range of plants and warehouses
    plants = range(len(capacity))
    warehouses = range(len(demand))

    # Model
    m = Model("facility")

    # Plant open decision variables: open[p] == 1 if plant p is open.
    open = []
    for p in plants:
        open.append(m.addVar(vtype=GRB.BINARY, name="Open%d" % p))

    # Transportation decision variables: how much to transport from
    # a plant p to a warehouse w
    transport = []
    for w in warehouses:
        transport.append([])
        for p in plants:
            transport[w].append(m.addVar(vtype=GRB.BINARY,
                                         name="Trans%d.%d" % (p, w)))

    # The objective is to minimize the total fixed and variable costs
    m.modelSense = GRB.MINIMIZE
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 600)

    # Update model to integrate new variables
    m.update()

    # Set optimization objective - minimize sum of fixed costs
    m.setObjective(quicksum([fixedCosts[p]*open[p] for p in plants]))

    for w in warehouses:
        m.setObjective(quicksum(transport[w][p] * transCosts[w][p] for p in plants))

    # Production constraints
    # Note that the right-hand limit sets the production to zero if the plant
    # is closed
    for p in plants:
        m.addConstr(
            quicksum(transport[w][p]*demand[w] for w in warehouses) <= capacity[p] * open[p],
            "Capacity%d" % p)

    # Demand constraints
    for w in warehouses:
        m.addConstr(quicksum(transport[w][p] for p in plants) == 1,
                    "Demand%d" % w)

    # Guess at the starting point: close the plant with the highest fixed costs;
    # open all others

    # First, open all plants
    for p in plants:
        open[p].start = 1.0

    # Now close the plant with the highest fixed cost
    # print('Initial guess:')
    maxFixed = max(fixedCosts)
    for p in plants:
        if fixedCosts[p] == maxFixed:
            open[p].start = 0.0
            # print('Closing plant %s' % p)
            break
    # print('')

    # Use barrier to solve root relaxation
    m.params.method = 2

    # Solve
    m.optimize()

    # Print solution
    solution = [-1]*len(customers)
    obj = m.objVal
    for w in warehouses:
        for p in plants:
            if transport[w][p].x > 0:
                solution[w] = p

    # print('\nTOTAL COSTS: %g' % m.objVal)
    # print('SOLUTION:')
    # for p in plants:
    #     if open[p].x == 1.0:
    #         print('Plant %s open' % p)
    #         for w in warehouses:
    #             if transport[w][p].x > 0:
    #                 print('  Transport %g units to warehouse %s' % \
    #                       (transport[w][p].x, w))
    #     else:
    #         print('Plant %s closed!' % p)
    #
    #
    #
    # solution = [-1]*len(customers)
    # capacity_remaining = [f.capacity for f in facilities]
    #
    # facility_index = 0
    # for customer in customers:
    #     if capacity_remaining[facility_index] >= customer.demand:
    #         solution[customer.index] = facility_index
    #         capacity_remaining[facility_index] -= customer.demand
    #     else:
    #         facility_index += 1
    #         assert capacity_remaining[facility_index] >= customer.demand
    #         solution[customer.index] = facility_index
    #         capacity_remaining[facility_index] -= customer.demand
    #
    # used = [0]*len(facilities)
    # for facility_index in solution:
    #     used[facility_index] = 1
    #
    # # calculate the cost of the solution
    # obj = sum([f.setup_cost*used[f.index] for f in facilities])
    # for customer in customers:
    #     obj += length(customer.location, facilities[solution[customer.index]].location)

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


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
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)'

