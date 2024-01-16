from 多深紧致化RMFS中机器人的任务调度.数值实验.MCRMFS_Instance_A import Instance_all
import gurobi_for_MCRMFS
import xlwt


book = xlwt.Workbook(encoding='utf-8')
sheet = book.add_sheet("gurobi_Solution")
# 初始化算例
small_Instance_Mairx = [
                        [1,1,3,4,1,1,0.2,None],
                        [1,1,3,6,1,1,0.2,None],
                        [1, 2, 3, 4, 1, 1, 0.2, None],
                        [1, 2, 3, 6, 1, 1, 0.2, None],
                        [2,2,3,4,1,1,0.2,None],
                        [2,2,3,6,1,1,0.2,None],
                        [2,3,3,4,1,1,0.2,None],
                        [2,3,3,6,1,1,0.2,None],
                        [3, 3, 3, 4, 1, 1, 0.2, None],
                        [3, 3, 3, 6, 1, 1, 0.2, None],

    [1, 1, 3, 4, 2, 1, 0.2, None],
    [1, 1, 3, 6, 2, 1, 0.2, None],
    [1, 2, 3, 4, 2, 1, 0.2, None],
    [1, 2, 3, 6, 2, 1, 0.2, None],
    [2, 2, 3, 4, 2, 1, 0.2, None],
    [2, 2, 3, 6, 2, 1, 0.2, None],
    [2, 3, 3, 4, 2, 1, 0.2, None],
    [2, 3, 3, 6, 2, 1, 0.2, None],


    [1, 1, 3, 3, 2, 2, 0.2, None], # 36
    [1, 1, 3, 4, 2, 2, 0.2, None],
    [1, 1, 3, 5, 2, 2, 0.2, None],
    [1, 1, 3, 6, 2, 2, 0.2, None],

    # 补充的

[1, 1, 3, 3, 3, 3, 0.5, None], # 22
[1, 1, 3, 4, 3, 3, 0.5, None],
[1, 1, 3, 5, 3, 3, 0.5, None],
[1, 1, 3, 6, 3, 3, 0.5, None],

[1, 1, 3, 3, 4, 4, 0.5, None], # 26
[1, 1, 3, 4, 4, 4, 0.5, None],
[1, 1, 3, 5, 4, 4, 0.5, None],
[1, 1, 3, 6, 4, 4, 0.5, None],

[2, 1, 3, 3, 2, 2, 0.5, None], # 30
[2, 1, 3, 4, 2, 2, 0.5, None],
[2, 1, 3, 5, 2, 2, 0.5, None],
[2, 1, 3, 6, 2, 2, 0.5, None],

[2, 1, 3, 3, 3, 3, 0.5, None], # 34
[2, 1, 3, 4, 3, 3, 0.5, None],
[2, 1, 3, 5, 3, 3, 0.5, None],
[2, 1, 3, 6, 3, 3, 0.5, None],

[2, 1, 3, 3, 4, 4, 0.5, None], # 38
[2, 1, 3, 4, 4, 4, 0.5, None],
[2, 1, 3, 5, 4, 4, 0.5, None],
[2, 1, 3, 6, 4, 4, 0.5, None],



                        ]
sheet.write(0, 0, "算例编号")
sheet.write(0, 1, "Obj")
sheet.write(80, 1, "Time")
# 迭代循环求解取平均
for i in range(22,len(small_Instance_Mairx)):
    sheet.write(i + 1, 0, i + 1)
    sheet.write(i + 81, 0, i + 1)
    for j in range(1):
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
        ## Gurobi求解
        Gurobi_Alg = gurobi_for_MCRMFS.gurobi_MCRMFS(problem=small_Instance, time_limit=3600)

        model, Obj, Time, objBound = Gurobi_Alg.run()
        print("第----------------------------------------------------------------------------------------------------------------",i+1,"组算例","    ",j+1,"次实验","   ",Obj, "   ",Time)
        # 输出本次求解结果
        sheet.write(i + 1, j+1, str(Obj))
        sheet.write(i + 1, j + 3, str(objBound))
        sheet.write(i + 81, j+1, str(Time))
        savepath = "D:\PythonProjet\SF-learning\多深紧致化RMFS中机器人的任务调度\MCRMFS_Gurobi\gurobi_Solution_1.xls"
        book.save(savepath)