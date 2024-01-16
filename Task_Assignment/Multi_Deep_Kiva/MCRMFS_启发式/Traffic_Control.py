# 系统交通碰撞的检测和修正算法 2023.02.21 by 626


# 机器人类，储存机器人的相关信息
class robot_traffic_control():
    def __init__(self):
        self.robot_route = [] # 列表存储机器人的路径
        self.robot_time = [] # 列表存储机器人路径对应的时间

# 冲突检测模块
class conflict_detection():
    pass

# 冲突修正模块
class conflict_correction():
    pass