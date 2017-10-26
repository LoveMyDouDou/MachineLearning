#coding=utf-8
import random

from FeatureSeletion.tools import load_data

# loop_condition=8#最起码要大于lifetime=15的值 ，因为播种一次age才曾1
# initialization_parameters = [15, 12, 30, 0.05, 50]

def read_in_trainset(file_dir,file_name):
    trainX,trainy=load_data(file_dir+file_name)#trainX,trainy are all list
    return trainX,trainy

def read_in_predictset(file_dir,file_name):
    predictx,predicty=load_data(file_dir+file_name)
    return predictx,predicty


def random_init(forest_area,tree_size):
    '''
    建立树的初始种群
    :param forest_area:
    :param tree_size:
    :return:
    '''
    init_forest=[]
    for i in range(0,forest_area):
        tree=[]
        for j in range(0,tree_size):
            tree.append(random.randint(0,1))
        init_forest.append(tree)
    return init_forest






