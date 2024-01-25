import gurobipy as gp
from Data2 import Data2

class TSP_model:
    def __init__(self , data):
        self.data = data
        
        
    def solve_route(self , route):
        disMatrix = self.update_dis(route)
        is_same_node = self.update_node(route)
        
        model = gp.Model("search_route")
        
        route_dict = {i : v for i , v in enumerate(route)}
        
        point_num = len(route) # 订单点数
        x = []
        t = []
        q = []
        
        for i in range(point_num + 2):
            x_tmp = []
            for j in range(point_num + 2):
                x_tmp.append(model.addVar(0,1,0,gp.GRB.BINARY,"x_%d_%d"%(i,j)))
            x.append(x_tmp)

            t.append(model.addVar(0 , 2500 , 0 , gp.GRB.CONTINUOUS , "t_%d"%i))     
            
            q.append(model.addVar(0 , 100 , 0 , gp.GRB.CONTINUOUS , "q_%d"%i))
        
        model.setObjective(gp.quicksum(disMatrix[i][j] * x[i][j] for i in range(point_num + 2) for j in range(point_num + 2)) , gp.GRB.MINIMIZE)
        
        model.addConstr(gp.quicksum(x[0][j] for j in range(1,point_num + 1)) == 1) # 虚拟起点出度为1
         
        model.addConstr(gp.quicksum(x[i][point_num + 1] for i in range(1,point_num + 1)) == 1) # 虚拟终点入度为1
        
        model.addConstr(gp.quicksum(x[i][0] for i in range(1,point_num + 2)) == 0) # 虚拟起点入度为0
        
        model.addConstr(gp.quicksum(x[point_num + 1][j] for j in range(point_num + 1)) == 0) # 虚拟终点出度为0
        
        model.addConstrs(gp.quicksum(x[i][j]   for j in range(point_num +2 ) if i != j) == 1  for i in range(1,point_num + 1))  # 每个中间点出度为1
        
        model.addConstrs(gp.quicksum(x[i][j]  for i in range(point_num +2 ) if i != j) == 1 for j in range(1,point_num + 1))  # 每个中间点入度为1
        
        model.addConstrs(q[i] + self.data.points[route_dict[i-1]].demand <= 100 for i in range(1,point_num + 1))
        
        model.addConstrs(q[i] + self.data.points[route_dict[i-1]].demand <= q[j] + 200 * (1 - x[i][j])  for i in range(1,point_num + 1) for j in range(1,point_num + 1) if i != j)
        
        model.addConstrs(t[i] <= 2400 for i in range(1,point_num + 1)) 
        
        model.addConstrs(t[i] >= self.data.points[route_dict[i-1]].time_window[0] for i in range(1,point_num + 1))
        
        model.addConstrs(t[i] + self.data.vehicle_service_time * is_same_node[i][j] + disMatrix[i][j] <= t[j] + 10e4 * (1 - x[i][j]) for i in range(1,point_num + 2) for j in range(point_num + 2) if i != j)
        
        for i in range(point_num ):
            if(self.data.points[route_dict[i]].type == 'P'):
                index_P = i
                index_D = route.index(route_dict[i] + self.data.order_num)
                model.addConstr(t[index_P+1] <= t[index_D+1])
        
        for i in range(point_num +1 ):
            x[i][i+1].start = 1
        
        model.setParam('TimeLimit', 10)
        model.update()
        # model.write("filename.lp")
        model.setParam('OutputFlag', 0)
        model.optimize()
        
        # for var in model.getVars():
        #     if(var.x > 0 and var.varName.startswith('x')):
        #         print(var.varName , var.x)
        # for var in model.getVars():
        #     if (var.VarName.startswith('t')):
        #         print(var.VarName , var.x)
        if model.getAttr("Status") == gp.GRB.OPTIMAL:
            new_route = self.decode_route(x , route_dict)
            # print(new_route)
            return new_route 
        else:
            return route
        
    def update_dis(self , route):
        disMatrix   = []
        point_num = len(route)
        for i in range(point_num + 2):
            disMatrix_tmp = []
            for j in range(point_num + 2):
                if( i == 0 or j == 0 or i == point_num + 1 or j == point_num + 1):
                    disMatrix_tmp.append(0)
                else:
                    node1_id = self.data.points[route[i-1]].node.node_id
                    node2_id = self.data.points[route[j-1]].node.node_id
                    disMatrix_tmp.append(self.data.node_time_matrix[node1_id][node2_id])
            disMatrix.append(disMatrix_tmp)
            
        return disMatrix
    
    def update_node(self , route):
        is_same_node = []
        point_num = len(route)
        for i in range(point_num + 2):
            is_same_node_tmp = []
            for j in range(point_num + 2):
                if( i == 0 or j == 0 or i == point_num + 1 or j == point_num + 1):
                    is_same_node_tmp.append(1)
                else:
                    if(self.data.points[route[i-1]].node.node_id == self.data.points[route[j-1]].node.node_id):
                        is_same_node_tmp.append(1)
                    else:
                        is_same_node_tmp.append(0)
            is_same_node.append(is_same_node_tmp)
            
        return is_same_node
    
    def decode_route(self , x , route_dict):
        point_num = len(route_dict)
        route = []
        start = 0
        for i in range(1,point_num + 1):
            if(x[0][i].x > 0.5):
                start = i
                break
        route.append(route_dict[start - 1])
        
        while (len(route) < point_num):
            for i in range(1,point_num + 1):
                if(x[start][i].x > 0.5):
                    route.append(route_dict[i-1])
                    start = i
                    break
        return route
    
if __name__ == "__main__":
    data = Data2(limited_order_num = 100)
    # route = [29,51,53,129,151,153,61,163,1,20,120,35,41,28,101,128,135,141,27,67,82,127,182,161,167]
    route = [51, 79, 61, 67, 179, 151, 161, 167]
    
    import Fitness
    print(Fitness.cal_car_fitness(data, route))
    alg = TSP_model(data)
    route = alg.solve_route(route)
    
    
    import Fitness
    print(Fitness.cal_car_fitness(data, route))
    