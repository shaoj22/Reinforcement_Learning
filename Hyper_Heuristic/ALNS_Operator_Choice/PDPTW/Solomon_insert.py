import numpy as np
from Sol import Sol

class solomon_insert_algorithm:
    def __init__(self , data):
        self.data = data
        
    def solomon_insert(self):
        # unvisit = self.data.P_set.copy()
        unvisit = self.data.points[1 : self.data.order_num + 1]
        visited = []
        route = []
        while(unvisit):
            tmpRoute = []
             
            
            selectPoint = self.findSelectPoint(unvisit)
            tmpRoute.append(selectPoint)
            tmpRoute.append(selectPoint+self.data.order_num)
            
            unvisit.remove(selectPoint)
            visited.append(selectPoint)

            flag = True
            startTime = [0]
            while(flag and unvisit):
                selectNearest = self.findNearestNode(self.data.points[tmpRoute[-1]].node , unvisit)
                time_travel = self.data.node_time_matrix[self.data.points[tmpRoute[-1]].node.node_id][self.data.points[selectNearest].node.node_id]
                time_1 = self.data.node_time_matrix[self.data.points[tmpRoute[-1]].node.node_id][self.data.points[tmpRoute[-2]].node.node_id]
                time_2 = self.data.node_time_matrix[self.data.points[selectNearest].node.node_id][self.data.points[selectNearest+self.data.order_num].node.node_id]
                time_now = startTime[-1] + time_1 +time_travel + time_2 + 4 * self.data.vehicle_service_time
                if(time_now < self.data.points[selectNearest+self.data.order_num].time_window[1]):
                    tmpRoute.append(selectNearest)
                    tmpRoute.append(selectNearest + self.data.order_num)
                    unvisit.remove(selectNearest)
                    visited.append(selectNearest)
                    startTime.append(startTime[-1] + time_1 +time_travel + 2 * self.data.vehicle_service_time)
                    if(len(unvisit) == 0):
                        flag = False
                        route.append(tmpRoute)
                else:
                    flag = False
                    route.append(tmpRoute)
                    
        return route
    
    def solve(self, solution:Sol):
        unvisit = self.data.P_set.copy()
        visited = []
        while(unvisit):
            tmpRoute = []
             
            
            selectPoint = self.findSelectPoint(unvisit)
            tmpRoute.append(selectPoint)
            tmpRoute.append(selectPoint+self.data.order_num)
            
            unvisit.remove(selectPoint)
            visited.append(selectPoint)

            flag = True
            startTime = [0]
            while(flag and unvisit):
                selectNearest = self.findNearestNode(self.data.points[tmpRoute[-1]].node , unvisit)
                time_travel = self.data.node_time_matrix[self.data.points[tmpRoute[-1]].node.node_id][self.data.points[selectNearest].node.node_id]
                time_1 = self.data.node_time_matrix[self.data.points[tmpRoute[-1]].node.node_id][self.data.points[tmpRoute[-2]].node.node_id]
                time_2 = self.data.node_time_matrix[self.data.points[selectNearest].node.node_id][self.data.points[selectNearest+self.data.order_num].node.node_id]
                time_now = startTime[-1] + time_1 +time_travel + time_2 + 4 * self.data.vehicle_service_time
                if(time_now < self.data.points[selectNearest+self.data.order_num].time_window[1]):
                    tmpRoute.append(selectNearest)
                    tmpRoute.append(selectNearest+self.data.order_num)
                    unvisit.remove(selectNearest)
                    visited.append(selectNearest)
                    startTime.append(startTime[-1] + time_1 +time_travel + 2 * self.data.vehicle_service_time)
                    if(len(unvisit) == 0):
                        flag = False
                        solution.add_route(tmpRoute)
                else:
                    flag = False
                    solution.add_route(tmpRoute)        
            
    
    def findSelectPoint(self , unvisit):
        return unvisit[0]
    
    def findNearestNode(self , node , candidateList):
        tmp = np.inf 
        select = -1
        for i in candidateList:
            if(self.data.node_time_matrix[node.node_id][self.data.points[i].node.node_id] < tmp):
                tmp = self.data.node_time_matrix[node.node_id][self.data.points[i].node.node_id]
                select = i
        return select