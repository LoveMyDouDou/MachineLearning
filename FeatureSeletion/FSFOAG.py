#coding=utf-8
import random

from FeatureSeletion.tools import load_data, num_to_feature, read_data_feature, train_knn, num_to_string, \
    string_to_numlist, num_to_list, train_svm, train_tree, calculate_DR, num_all_zero


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

def one_point_hybridization_knn(forest_list,feature_list,trainx,trainy,predictx,predicty,neighbor):
    '''
    一次单点杂交
    :param forest_list: 森林列表，双层列表
    :param feature_list:特征集合索引,特征集合的角标
    :param trainx:训练集
    :param trainy:训练集对应的分类
    :param predictx:预测集合
    :param predicty:预测集对应的分类
    :return:森林字典:包括单点杂交后每棵树的准确率和森林的01串
            森林列表:单点杂交后新的森林
    '''
    forest = {}  # 记录森林里的准确率
    forest_list=one_point_hybridization(forest_list)
    for num in forest_list:
        feature=num_to_feature(num,feature_list)
        train_sample=read_data_feature(feature,trainx)
        predict_sample=read_data_feature(feature,predictx)
        acc = train_knn(train_sample, trainy, predict_sample, predicty,neighbor)
        num_string = num_to_string(num)
        forest[num_string] = acc
    return forest,forest_list

def one_point_hybridization_knn_result(forest_old, feature_list, trainx, trainy, predictx, predicty,loop,neighbor):
    res = {}
    for i in range(0, loop):
        forest, forest_new = one_point_hybridization_knn(forest_old, feature_list, trainx, trainy, predictx, predicty,neighbor)
        forest_list = sorted(forest.items(), key=lambda item:(item[1],item[0]), reverse=True)
        forest_next = []
        for j in range(0, 10):
            forest_next.append(forest_old[j])
        for j in range(0, 30):
            forest_next.append(num_to_list(forest_list[j][0]))
        gene_mutation_list = []
        for j in range(30, 40):
            gene_mutation_list.append(num_to_list(forest_list[j][0]))
        positive_table, nagative_table = calculate_table_knn(gene_mutation_list, feature_list, trainx, trainy, predictx,
                                                             predicty,neighbor)
        for j in range(0, 10):
            forest_next.append(gene_mutation(positive_table, nagative_table))

        forest_old = forest_next
        # res.append(forest_list[0][1])
        res[forest_list[0][0]]=forest_list[0][1]
    first=sorted(res.items(), key=lambda item:(item[1],item[0]),reverse=True)[0]
    return first



def gene_mutation(positive_table,nagative_table):
    '''
    给定正表和负表，产生新的基因
    :param positive_table: 正表，里面存储的应是整型变量
    :param nagative_table: 负表，里面存储的应是整型变量
    :return: 一个基因列表，里面存储的是数值0和1
    '''
    list=[]
    for i in range(0,len(positive_table)):
        a=positive_table[i]
        b=nagative_table[i]
        c=a+b
        r=random.randint(1,c)
        if r<=a:
            list.append(1)
        else:
            list.append(0)
    return list

def reverse_index(num,index):
    '''
    将列表第index位取反，0变1,1变0
    :param num:
    :param index:
    :return:
    '''
    list=[]
    for i in range(0,len(num)):
        if i!=index:
            list.append(num[i])
        else:
            if num[i]==1:
                list.append(0)
            else:
                list.append(1)
    return list

def calculate_table_knn(str_list,feature_list, trainx, trainy, predictx, predicty,neighbor):
    positive_table=[]
    nagative_table=[]
    length=len(str_list[0])
    for i in range(0,length):
        positive_table.append(len(str_list))
        nagative_table.append(len(str_list))
    for num in str_list:
        feature=num_to_feature(num,feature_list)
        train_sample=read_data_feature(feature,trainx)
        predict_sample=read_data_feature(feature,predictx)
        acc_original = train_knn(train_sample, trainy, predict_sample, predicty,neighbor)
        for i in range(0,length):
            new_num=reverse_index(num,i)
            feature = num_to_feature(new_num, feature_list)
            train_sample = read_data_feature(feature, trainx)
            predict_sample = read_data_feature(feature, predictx)
            acc_new= train_knn(train_sample, trainy, predict_sample, predicty,neighbor)
            if acc_new>acc_original:
                if num[i]==0:
                    nagative_table[i]+=1
                else:
                    positive_table[i]+=1
            else:
                if num[i]==1:
                    nagative_table[i]+=1
                else:
                    positive_table[i]+=1
    return positive_table,nagative_table


def one_point_hybridization_svm_result(forest_old, feature_list, trainx, trainy, predictx, predicty,loop):
    res = {}
    for i in range(0, loop):
        forest, forest_new = one_point_hybridization_svm(forest_old, feature_list, trainx, trainy, predictx, predicty)
        forest_list = sorted(forest.items(), key=lambda item:(item[1],item[0]), reverse=True)
        forest_next = []
        for j in range(0, 10):
            forest_next.append(forest_old[j])
        for j in range(0, 30):
            forest_next.append(num_to_list(forest_list[j][0]))
        gene_mutation_list = []
        for j in range(30, 40):
            gene_mutation_list.append(num_to_list(forest_list[j][0]))
        positive_table, nagative_table = calculate_table_svm(gene_mutation_list, feature_list, trainx, trainy, predictx,
                                                             predicty)
        for j in range(0, 10):
            forest_next.append(gene_mutation(positive_table, nagative_table))

        forest_old = forest_next
        # res.append(forest_list[0][1])
        res[forest_list[0][0]]=forest_list[0][1]
    first=sorted(res.items(), key=lambda item:(item[1],item[0]),reverse=True)[0]
    return first

def one_point_hybridization_svm(forest_list,feature_list,trainx,trainy,predictx,predicty):
    '''
    一次单点杂交
    :param forest_list: 森林列表，双层列表
    :param feature_list:特征集合索引,特征集合的角标
    :param trainx:训练集
    :param trainy:训练集对应的分类
    :param predictx:预测集合
    :param predicty:预测集对应的分类
    :return:森林字典:包括单点杂交后每棵树的准确率和森林的01串
            森林列表:单点杂交后新的森林
    '''
    forest = {}  # 记录森林里的准确率
    forest_list=one_point_hybridization(forest_list)
    for num in forest_list:
        feature=num_to_feature(num,feature_list)
        train_sample=read_data_feature(feature,trainx)
        predict_sample=read_data_feature(feature,predictx)
        acc = train_svm(train_sample, trainy, predict_sample, predicty)
        num_string = num_to_string(num)
        forest[num_string] = acc
    return forest,forest_list

def calculate_table_svm(str_list,feature_list, trainx, trainy, predictx, predicty):
    positive_table=[]
    nagative_table=[]
    length=len(str_list[0])
    for i in range(0,length):
        positive_table.append(len(str_list))
        nagative_table.append(len(str_list))
    for num in str_list:
        if num_all_zero(num):
            num[0]=1
        feature=num_to_feature(num,feature_list)
        train_sample=read_data_feature(feature,trainx)
        predict_sample=read_data_feature(feature,predictx)
        acc_original = train_svm(train_sample, trainy, predict_sample, predicty)
        for i in range(0,length):
            new_num=reverse_index(num,i)
            feature = num_to_feature(new_num, feature_list)
            train_sample = read_data_feature(feature, trainx)
            predict_sample = read_data_feature(feature, predictx)
            acc_new= train_svm(train_sample, trainy, predict_sample, predicty)
            if acc_new>acc_original:
                if num[i]==0:
                    nagative_table[i]+=1
                else:
                    positive_table[i]+=1
            else:
                if num[i]==1:
                    nagative_table[i]+=1
                else:
                    positive_table[i]+=1
    return positive_table,nagative_table


def one_point_hybridization_train_tree_result(forest_old, feature_list, trainx, trainy, predictx, predicty,loop):
    res = {}
    for i in range(0, loop):
        forest, forest_new = one_point_hybridization_train_tree(forest_old, feature_list, trainx, trainy, predictx, predicty)
        forest_list = sorted(forest.items(), key=lambda item:(item[1],item[0]), reverse=True)
        forest_next = []
        for j in range(0, 10):
            forest_next.append(forest_old[j])
        for j in range(0, 30):
            forest_next.append(num_to_list(forest_list[j][0]))
        gene_mutation_list = []
        for j in range(30, 40):
            gene_mutation_list.append(num_to_list(forest_list[j][0]))
        positive_table, nagative_table = calculate_table_train_tree(gene_mutation_list, feature_list, trainx, trainy, predictx,
                                                             predicty)
        for j in range(0, 10):
            forest_next.append(gene_mutation(positive_table, nagative_table))

        forest_old = forest_next
        # res.append(forest_list[0][1])
        res[forest_list[0][0]]=forest_list[0][1]
    first=sorted(res.items(), key=lambda item:(item[1],item[0]),reverse=True)[0]
    return first


def one_point_hybridization_train_tree(forest_list,feature_list,trainx,trainy,predictx,predicty):
    '''
    一次单点杂交
    :param forest_list: 森林列表，双层列表
    :param feature_list:特征集合索引,特征集合的角标
    :param trainx:训练集
    :param trainy:训练集对应的分类
    :param predictx:预测集合
    :param predicty:预测集对应的分类
    :return:森林字典:包括单点杂交后每棵树的准确率和森林的01串
            森林列表:单点杂交后新的森林
    '''
    forest = {}  # 记录森林里的准确率
    forest_list=one_point_hybridization(forest_list)
    for num in forest_list:
        feature=num_to_feature(num,feature_list)
        train_sample=read_data_feature(feature,trainx)
        predict_sample=read_data_feature(feature,predictx)
        acc = train_tree(train_sample, trainy, predict_sample, predicty)
        num_string = num_to_string(num)
        forest[num_string] = acc
    return forest,forest_list

def calculate_table_train_tree(str_list,feature_list, trainx, trainy, predictx, predicty):
    positive_table=[]
    nagative_table=[]
    length=len(str_list[0])
    for i in range(0,length):
        positive_table.append(len(str_list))
        nagative_table.append(len(str_list))
    for num in str_list:
        if num_all_zero(num):
            num[0]=1
        feature=num_to_feature(num,feature_list)
        train_sample=read_data_feature(feature,trainx)
        predict_sample=read_data_feature(feature,predictx)
        acc_original = train_tree(train_sample, trainy, predict_sample, predicty)
        for i in range(0,length):
            new_num=reverse_index(num,i)
            feature = num_to_feature(new_num, feature_list)
            train_sample = read_data_feature(feature, trainx)
            predict_sample = read_data_feature(feature, predictx)
            acc_new= train_tree(train_sample, trainy, predict_sample, predicty)
            if acc_new>acc_original:
                if num[i]==0:
                    nagative_table[i]+=1
                else:
                    positive_table[i]+=1
            else:
                if num[i]==1:
                    nagative_table[i]+=1
                else:
                    positive_table[i]+=1
    return positive_table,nagative_table


def DO_FSFOA(file_dir_name,predictset_file_dir, predictset_file_name,trainx,trainy,loop):
    predictx, predicty=read_in_predictset(predictset_file_dir, predictset_file_name)

    feature_list = []  # 特征集合索引,特征集合的角标
    for i in range(0,len(trainx[0])):
        feature_list.append(i)
    forest={}  #记录森林里的准确率
    init_forest=random_init(50,len(trainx[0]))

    write_to=file_dir_name+'_'+predictset_file_name+'_result.txt'
    file_object = open(write_to, 'a')
    file_object.write(predictset_file_name+':\n')

    for neighbor in range(1,6,2):
        tree,acc=one_point_hybridization_knn_result(init_forest, feature_list, trainx, trainy, predictx, predicty,loop,neighbor)
        file_object.write('knn n=')
        file_object.write(str(neighbor))
        file_object.write('\n')
        file_object.write('accuracy=')
        file_object.write(str(acc))
        file_object.write('\n')
        file_object.write('DR=')
        file_object.write(str(calculate_DR(tree)))
        file_object.write('\n')

    tree,acc=one_point_hybridization_svm_result(init_forest, feature_list, trainx, trainy, predictx, predicty,loop)
    # print 'svm'
    # print 'accuracy=',acc
    # print 'DR=',calculate_DR(tree)
    file_object.write('svm')
    file_object.write('\n')
    file_object.write('accuracy=')
    file_object.write(str(acc))
    file_object.write('\n')
    file_object.write('DR=')
    file_object.write(str(calculate_DR(tree)))
    file_object.write('\n')

    tree,acc=one_point_hybridization_train_tree_result(init_forest, feature_list, trainx, trainy, predictx, predicty, loop)
    # print 'tree'
    # print 'accuracy=', acc
    # print 'DR=', calculate_DR(tree)
    file_object.write('tree')
    file_object.write('\n')
    file_object.write('accuracy=')
    file_object.write(str(acc))
    file_object.write('\n')
    file_object.write('DR=')
    file_object.write(str(calculate_DR(tree)))
    file_object.write('\n')

    file_object.write('----------------------------------------')
    file_object.write('\n')
    file_object.close()


