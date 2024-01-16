"""
测试接口文件
"""
import sys
sys.path.append("..")
import main as runner
import xlwt
from MCRMFS_Instance_A_Large import Instance_all

def experiment(instance, render=False, method="heuristic", test_times=1):
    """experiment

    Args:
        instance (_type_): 测试算例
        render (bool, optional): 是否展示过程. Defaults to False.
        method (str, optional): 测试什么方法. Defaults to "heuristic". option: "net" or "heuristic"
        test_times (int, optional): 测试多少次取平均 (每次随机种子从0到test_times-1). Defaults to 10.
    """
    args = runner.args.copy()
    args["instance"] = instance
    args["render"] = render
    args["method"] = method
    args["test_size"] = test_times
    rewards, timecost = runner.train(args)
    avg_reward = sum(rewards) / len(rewards)
    avg_finish_num = round(avg_reward / 1000, 2)
    avg_step_num = round((1 - avg_reward % 1) / 0.0000001, 2)
    print("平均完成目标数：{:.2f}, 平均步数：{:.2f}, 平均用时：{:.5f}s".format(avg_finish_num, avg_step_num, timecost))
    return avg_finish_num, avg_step_num, timecost



# 启发式求解
book = xlwt.Workbook(encoding='utf-8')
sheet = book.add_sheet("H_Solution")
# 初始化算例
small_Instance_Mairx = [
                        [2,6,3,3,3,20,0.2,None], # 1
                        [2,8,3,3,3,30,0.2,None],
                        [2, 10, 3, 3, 4, 40, 0.2, None],
                        [2, 12, 3, 3, 4, 50, 0.2, None],
                        [2,14,3,3,5,60,0.2,None],
                        [2,16,3,3,5,70,0.2,None],


                        [3,6,4,4,3,30,0.2,None],  # 6
                        [3,8,4,4,3,40,0.2,None],
                        [3, 10, 4, 4, 4, 50, 0.2, None],
                        [3, 12, 4, 4, 4, 60, 0.2, None],
                        [3, 14, 4, 4, 5, 70, 0.2, None],
                        [3, 16, 4, 4, 5, 80, 0.2, None],

    [4, 6, 5, 5, 3, 40, 0.2, None],  # 12
    [4, 8, 5, 5, 3, 50, 0.2, None],
    [4, 10, 5, 5, 4, 60, 0.2, None],
    [4, 12, 5, 5, 4, 70, 0.2, None],
    [4, 14, 5, 5, 5, 80, 0.2, None],
    [4, 16, 5, 5, 5, 90, 0.2, None],


    [5, 6, 6, 6, 3, 50, 0.2, None], # 18
    [5, 8, 6, 6, 3, 60, 0.2, None],
    [5, 10, 6, 6, 4, 70, 0.2, None],
    [5, 12, 6, 6, 4, 80, 0.2, None],
    [5, 14, 6, 6, 5, 90, 0.2, None],
    [5, 16, 6, 6, 5, 100, 0.2, None],

    [6, 6, 7, 7, 3, 60, 0.2, None],  # 24
    [6, 8, 7, 7, 3, 70, 0.2, None],
    [6, 10, 7, 7, 4, 80, 0.2, None],
    [6, 12, 7, 7, 4, 90, 0.2, None],
    [6, 14, 7, 7, 5, 100, 0.2, None],
    [3, 50, 3, 5, 100, 100, 0.2, None],
                        ]
sheet.write(0, 0, "算例编号")
sheet.write(0, 1, "Obj")
sheet.write(40,1, "Time")

for i in range(len(small_Instance_Mairx)-1, len(small_Instance_Mairx)):
    sheet.write(i + 1, 0, i + 1)
    sheet.write(i + 41, 0, i + 1)
    # 初始化算例和算法
    w_num = small_Instance_Mairx[i][0]
    l_num = small_Instance_Mairx[i][1]
    m = small_Instance_Mairx[i][2]
    n = small_Instance_Mairx[i][3]
    R = small_Instance_Mairx[i][4]
    num_g = small_Instance_Mairx[i][5]
    percent_e = small_Instance_Mairx[i][6]
    seed = 0
    # 初始化算例
    small_Instance = Instance_all(w_num, l_num, m, n, R, num_g, percent_e, seed)
    # 利用启发式求解
    avg_finish_num, avg_step_num, time_cost = experiment(small_Instance, test_times=1)
    print(time_cost)
    sheet.write(i + 1, 1, str(avg_step_num))
    sheet.write(i + 41, 1, str(time_cost))
    savepath = "D:\PythonProjet\SF-learning\多深紧致化RMFS中机器人的任务调度\数值实验\中大规模实验-强化学习.xls"
    book.save(savepath)

# if __name__ == "__main__":
#     instance = Instance_all()
#     experiment(instance, render = False, test_times=10)
