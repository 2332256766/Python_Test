import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1,1],[1,2],[2,1],[2,2],[4,4],[4,5],[5,4],[5,5]])
y = np.array([0,0,0,0,1,1,1,1])
t_x =  np.array([[3,3]])

def distance(x, y, test_x):
    '''1.计算各点的距离'''
    distance_list =[]
    for index,dot_data in enumerate(x):
        l = ((dot_data[0]-test_x[0,0])**2+(dot_data[1]-test_x[0,1])**2)**(1/2)
        distance_list.append([l,y[index]])
        # distance_list.append([y[index],l])
    distance_list.sort()
    print(distance_list)

    return distance_list


def choose_dot():
    '''选最近k个点，分别比较概率'''
    lst = distance(x, y,t_x)
    new_lst = [n[1] for n in lst[0:5]]
    a = new_lst.count(0) ;b = new_lst.count(1)
    if a<=b:
        print(b)
        return 1
    else:
        print(a)
        return 0

def run():
    result = choose_dot()
    print('预测值为',result)

if __name__ == '__main__':
    run()
