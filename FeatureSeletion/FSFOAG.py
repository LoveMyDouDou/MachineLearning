#coding=utf-8
import random

from FeatureSeletion.tools import load_data, num_to_feature, read_data_feature, train_knn, num_to_string


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
    :return:返回双层列表
    '''
    init_forest=[]
    for i in range(0,forest_area):
        tree=[]
        for j in range(0,tree_size):
            tree.append(random.randint(0,1))
        init_forest.append(tree)
    return init_forest


def one_point_hybridization1(str1,str2):
    '''
    单点杂交，输入两条染色体，产生新染色体
    :param str1: 给定的一条染色体
    :param str2: 给定的另一条染色体
    :return: 杂交后产生的两条新染色体
    '''
    index=random.randint(0,len(str1)-1)
    a=[]
    b=[]
    for i in range(0,index):
        a.append(str1[i])
        b.append(str2[i])
    for i in range(index,len(str1)):
        a.append(str2[i])
        b.append(str1[i])
    return a,b


def one_point_hybridization(strlist):
    '''
    单点杂交
    :param strlist:准备杂交的列表
    :return: 单点杂交后产生的列表
    '''
    new_stirng=[]
    if (len(strlist)%2==1):
        new_stirng.append(strlist[len(strlist)-1])
    for i in range(0,len(strlist),2):
        a,b=one_point_hybridization1(strlist[i],strlist[i+1])
        new_stirng.append(a)
        new_stirng.append(b)
    return new_stirng


def one_point_hybridization_knn(forest_list,feature_list,trainx,trainy,predictx,predicty):
    forest = {}  # 记录森林里的准确率
    forest_list=one_point_hybridization(forest_list)
    for num in forest_list:
        feature=num_to_feature(num,feature_list)
        train_sample=read_data_feature(feature,trainx)
        predict_sample=read_data_feature(feature,predictx)
        acc = train_knn(train_sample, trainy, predict_sample, predicty)
        num_string = num_to_string(num)
        forest[num_string] = acc
    return forest,forest_list












