"""
测试接口文件
"""
import sys
sys.path.append("D:\PythonProjet\SF-learning\多深紧致化RMFS中机器人的任务调度\kiva_RL")
import main as runner
import xlwt
from MCRMFS_Instance_A_Large import Instance_all

def experiment(instance, render=False, method="net", test_times=1):
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


[3, 3, 3, 3, 5, 10, 0.2, None],
[3, 3, 3, 4, 5, 10, 0.2, None],
[3, 3, 3, 5, 5, 10, 0.2, None],
[3, 3, 3, 6, 5, 10, 0.2, None],
[3, 3, 3, 7, 5, 10, 0.2, None],
[3, 3, 3, 8, 5, 10, 0.2, None],
[3, 3, 3, 9, 5, 10, 0.2, None],
[3, 3, 3, 10, 5, 10, 0.2, None],

[3, 3, 4, 3, 5, 10, 0.2, None],
[3, 3, 4, 4, 5, 10, 0.2, None],
[3, 3, 4, 5, 5, 10, 0.2, None],
[3, 3, 4, 6, 5, 10, 0.2, None],
[3, 3, 4, 7, 5, 10, 0.2, None],
[3, 3, 4, 8, 5, 10, 0.2, None],
[3, 3, 4, 9, 5, 10, 0.2, None],
[3, 3, 4, 10, 5, 10, 0.2, None],

[3, 3, 5, 3, 5, 10, 0.2, None],
[3, 3, 5, 4, 5, 10, 0.2, None],
[3, 3, 5, 5, 5, 10, 0.2, None],
[3, 3, 5, 6, 5, 10, 0.2, None],
[3, 3, 5, 7, 5, 10, 0.2, None],
[3, 3, 5, 8, 5, 10, 0.2, None],
[3, 3, 5, 9, 5, 10, 0.2, None],
[3, 3, 5, 10, 5, 10, 0.2, None],

[3, 3, 6, 3, 5, 10, 0.2, None],
[3, 3, 6, 4, 5, 10, 0.2, None],
[3, 3, 6, 5, 5, 10, 0.2, None],
[3, 3, 6, 6, 5, 10, 0.2, None],
[3, 3, 6, 7, 5, 10, 0.2, None],
[3, 3, 6, 8, 5, 10, 0.2, None],
[3, 3, 6, 9, 5, 10, 0.2, None],
[3, 3, 6, 10, 5, 10, 0.2, None],

[3, 3, 7, 3, 5, 10, 0.2, None],
[3, 3, 7, 4, 5, 10, 0.2, None],
[3, 3, 7, 5, 5, 10, 0.2, None],
[3, 3, 7, 6, 5, 10, 0.2, None],
[3, 3, 7, 7, 5, 10, 0.2, None],
[3, 3, 7, 8, 5, 10, 0.2, None],
[3, 3, 7, 9, 5, 10, 0.2, None],
[3, 3, 7, 10, 5, 10, 0.2, None],

[3, 3, 8, 3, 5, 10, 0.2, None],
[3, 3, 8, 4, 5, 10, 0.2, None],
[3, 3, 8, 5, 5, 10, 0.2, None],
[3, 3, 8, 6, 5, 10, 0.2, None],
[3, 3, 8, 7, 5, 10, 0.2, None],
[3, 3, 8, 8, 5, 10, 0.2, None],
[3, 3, 8, 9, 5, 10, 0.2, None],
[3, 3, 8, 10, 5, 10, 0.2, None],

[3, 3, 9, 3, 5, 10, 0.2, None],
[3, 3, 9, 4, 5, 10, 0.2, None],
[3, 3, 9, 5, 5, 10, 0.2, None],
[3, 3, 9, 6, 5, 10, 0.2, None],
[3, 3, 9, 7, 5, 10, 0.2, None],
[3, 3, 9, 8, 5, 10, 0.2, None],
[3, 3, 9, 9, 5, 10, 0.2, None],
[3, 3, 9, 10, 5, 10, 0.2, None],

[3, 3, 10, 3, 5, 10, 0.2, None],
[3, 3, 10, 4, 5, 10, 0.2, None],
[3, 3, 10, 5, 5, 10, 0.2, None],
[3, 3, 10, 6, 5, 10, 0.2, None],
[3, 3, 10, 7, 5, 10, 0.2, None],
[3, 3, 10, 8, 5, 10, 0.2, None],
[3, 3, 10, 9, 5, 10, 0.2, None],
[3, 3, 10, 10, 5, 10, 0.2, None],

                        ]
sheet.write(0, 0, "算例编号")
sheet.write(0, 1, "Obj")
sheet.write(80, 1, "Time")

for i in range(len(small_Instance_Mairx)):
    sheet.write(i + 1, 0, i + 1)
    sheet.write(i + 81, 0, i + 1)
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
    avg_finish_num, avg_step_num, time_cost = experiment(small_Instance, test_times=10)
    sheet.write(i + 1, 1, str(avg_step_num))
    sheet.write(i + 81,1, str(time_cost))
    savepath = "D:\PythonProjet\SF-learning\多深紧致化RMFS中机器人的任务调度\数值实验\数值实验8_2.xls"
    book.save(savepath)

# if __name__ == "__main__":
#     instance = Instance_all()
#     experiment(instance, render = False, test_times=1)
