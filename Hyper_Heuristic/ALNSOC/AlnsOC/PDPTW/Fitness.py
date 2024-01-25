import numpy as np
from Sol import Sol
from Data import Node
from Data2 import Data2


# def cal_car_fitness(data , route):
#     time_penalty = 1000
#     overload_penalty = 1000
    
#     startTime = 0
#     endTime = 2500
#     if (len(route) == 0):
#         return 0
#     cost = data.vehicle_fixed_cost
    
#     cap = [ 0 for i in range(len(route))]
#     time = [ 0 for i in range(len(route))]
    
#     for i in range(len(route)) :
#         if(i == 0):
#             cap[i] = data.points[route[i]].order.quantity
#             time[i] = 0
#         else:
#             cap[i] = cap[i-1] + data.points[route[i]].demand
#             travel_time = data.node_time_matrix[data.points[route[i-1]].node.node_id][data.points[route[i]].node.node_id] 
#             service_time = data.vehicle_service_time if data.points[route[i-1]].node.node_id != data.points[route[i]].node.node_id else 0
#             time[i] = time[i-1]  +   data.vehicle_service_time + travel_time
#             time[i] = min(time[i] , data.points[route[i]].time_window[0])
#             cost += data.vehicle_unit_travel_cost * travel_time
#             cost += time_penalty if time[i] > data.points[route[i]].time_window[1] else 0
#             cost += overload_penalty if cap[i] > data.vehicle_capacity else 0
    
#     return cost

# @profile
def cal_car_fitness(data, route):
    time_penalty = 1e5 
    overload_penalty = 1e5

    if len(route) == 0:
        return 0

    cap = np.zeros(len(route))
    time = np.zeros(len(route))
    cap[0] = data.points[route[0]].order.quantity
    time[0] = 0

    travel_times = np.array([data.node_time_matrix[data.points[route[i]].node.node_id][data.points[route[i+1]].node.node_id] for i in range(len(route) - 1)])
    service_times = np.array([data.vehicle_service_time if data.points[route[i]].node.node_id != data.points[route[i+1]].node.node_id else 0 for i in range(len(route) - 1)])
    demands = np.array([data.points[r].demand for r in route])
    cap[1:] = np.cumsum(demands[1:]) + cap[0]
    time[1:] = np.cumsum(travel_times + service_times) + time[0]
    
    time_window_start = np.array([data.points[r].time_window[0] for r in route[1:]])
    time_window_end = np.array([data.points[r].time_window[1] for r in route[1:]])

    time[1:] = np.maximum(time[1:], time_window_start)

    cost = data.vehicle_fixed_cost + np.sum(data.vehicle_unit_travel_cost * travel_times)
    cost += np.sum(time_penalty * (time[1:] +100 > time_window_end))
    cost += np.sum(overload_penalty * (cap > data.vehicle_capacity))

    time_count = np.count_nonzero(time[1:] + 100 > time_window_end)
    overload_count = np.count_nonzero(cap > data.vehicle_capacity)
    return cost
    # return cost , time_count , overload_count


def cal_fitness(data , sol):
    sol.objs = [0 for i in range(len(sol.routes))]
    for i in range(len(sol.routes)):
        sol.objs[i] = cal_car_fitness(data , sol.routes[i])
    
    sol.obj = sum(sol.objs)
    return sol.obj

def cal_fitness_changed(data , sol , changed_car_list):
    for route_index in changed_car_list:
        sol.objs[route_index] = cal_car_fitness(data , sol.routes[route_index])
    
    sol.obj = sum(sol.objs)
    return sol.obj
            