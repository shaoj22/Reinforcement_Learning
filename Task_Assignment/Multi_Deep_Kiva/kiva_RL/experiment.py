""" 
测试接口文件
"""
import main as runner
import math

from MCRMFS_Instance_A import Instance_all

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

if __name__ == "__main__":
    instance = Instance_all()
    experiment(instance, method="heuristic", test_times=10)
    experiment(instance, method="net", test_times=10)




