from Data2 import Data2
from ALNS_1 import *

if __name__ == '__main__':
    # test = [30,40,50]
    # for scale in test:
    #     iter_num = 1000
    #     location = "城市配送系统优化资料包\\输入数据"
    #     # data = Data2(scale , limited_order_num= 100 )
    #     data = Data2(location , scale)
    #     alg = ALNS(data, iter_num ,scale)
    #     alg.run()
    #     alg.plot_routes()
    #     alg.show_process()
    #     alg.save_data()
    #     alg.plot_all_route()
    
    test2 = [3,4,5,6,7,50]
    for scale in test2:
        iter_num = 1000
        location = "小算例\\小算例"
        data = Data2(location , scale)
        alg = ALNS(data, iter_num ,scale)
        alg.run()
        alg.save_data()