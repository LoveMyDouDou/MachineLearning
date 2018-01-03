#coding=utf-8
import os

from FeatureSeletion.FSFOAG import read_in_trainset, read_in_predictset, random_init, one_point_hybridization, \
    one_point_hybridization_knn, one_point_hybridization_knn_result, calculate_table_knn, gene_mutation, \
    one_point_hybridization_svm_result, one_point_hybridization_train_tree_result, DO_FSFOA
from FeatureSeletion.GFAFOA_HIGH import DO_FSFOA_HIGH
from FeatureSeletion.tools import train_svm, train_knn, train_tree, num_to_feature, read_data_feature, num_to_string, \
    string_to_numlist, num_to_list, calculate_DR





if __name__=='__main__':
    g = os.walk("C:\processed_data")

    file_dir_list = []
    for path, d, filelist in g:
        file_dir_list.append(path)
    # for dir in file_dir_list:
    #     print dir

    for i in range(1, len(file_dir_list)):
        for text_index in range(1,11):
            # print file_dir_list[i]
            s = str(file_dir_list[i]).split("\\")
            file_dir_name=s[2]
            # 训练集
            trainset_file_dir = file_dir_list[i]+'\\'
            trainset_file_dir = trainset_file_dir.replace('\\', '/')
            # trainset_file_name = 'train_'+str(text_index)+"_2fold.txt"
            # trainset_file_name = 'train_'+str(text_index)+".txt"
            trainset_file_name = 'train_70'+".txt"
            trainx, trainy = read_in_trainset(trainset_file_dir, trainset_file_name)
            # 预测集
            predictset_file_dir = file_dir_list[i]+'\\'
            predictset_file_dir = predictset_file_dir.replace('\\', '/')
            # predictset_file_name = 'predict_'+str(text_index)+".txt"
            # predictset_file_name = 'predict_'+str(text_index)+"_2fold.txt"
            predictset_file_name = 'predict_30'+".txt"
            print predictset_file_dir
            print predictset_file_name
            # DO_FSFOA(file_dir_name,predictset_file_dir, predictset_file_name, trainx, trainy, 50)
            DO_FSFOA_HIGH(file_dir_name,predictset_file_dir, predictset_file_name, trainx, trainy, 100)
    print 'updating...'


    # #训练集
    # trainset_file_dir = 'C:\processed_data\sonar\\'
    # trainset_file_dir = trainset_file_dir.replace('\\', '/')
    # trainset_file_name='train_1.txt'
    # trainx,trainy=read_in_trainset(trainset_file_dir, trainset_file_name)
    # #预测集
    # predictset_file_dir = 'C:\processed_data\sonar\\'
    # predictset_file_dir = predictset_file_dir.replace('\\', '/')
    # predictset_file_name = 'predict_1.txt'

    # DO_FSFOA(predictset_file_dir, predictset_file_name,trainx,trainy,2)
    print 'execute end'
















