"""
测试接口文件
"""
import main as runner
import xlwt
from 多深紧致化RMFS中机器人的任务调度.kiva_RL.MCRMFS_Instance_A_Large import Instance_all

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
    reward, timecost = runner.train(args)[0]
    avg_finish_num = round(reward / 1000, 2)
    avg_step_num = round((1 - reward % 1) / 0.000001, 2)
    print("平均完成目标数：{:.2f}, 平均步数：{:.2f}, 平均用时：{:.5f}s".format(avg_finish_num, avg_step_num, timecost))
    return avg_finish_num, avg_step_num, timecost



# 启发式求解
book = xlwt.Workbook(encoding='utf-8')
sheet = book.add_sheet("H_Solution")
# 初始化算例
small_Instance_Mairx = [
[2,8,3,3,5,6,0.2,None],
[2,8,3,3,5,6,0.2,None],
[2,8,3,3,5,6,0.2,None],# 1
[3,8,4,4,5,6,0.2,None],
[3,8,4,4,5,6,0.2,None,2],
[3,8,4,4,5,6,0.2,None,3],
[3,8,4,4,5,6,0.2,None,4],# 2
[4,8,4,4,5,6,0.2,None,1],
[4,8,4,4,5,6,0.2,None,2],
[4,8,4,4,5,6,0.2,None,3],
[4,8,4,4,5,6,0.2,None,4],
[4,8,4,4,5,6,0.2,None,5],# 3
[5,8,6,6,5,6,0.2,None],
[6,8,7,7,5,6,0.2,None],
                        ]
sheet.write(0, 0, "算例编号")
sheet.write(0, 1, "Obj")
sheet.write(40, 1, "Time")

for i in range(6,11):
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
    sheet.write(i + 1, 1, str(avg_step_num))
    sheet.write(i + 41,1, str(time_cost))
    savepath = "D:\PythonProjet\SF-learning\多深紧致化RMFS中机器人的任务调度\kiva_RL\数值实验2.xls"
    book.save(savepath)

# if __name__ == "__main__":
#     instance = Instance_all()
#     experiment(instance, render = False, test_times=1)
