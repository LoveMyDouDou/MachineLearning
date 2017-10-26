#coding=utf-8
from FeatureSeletion.FSFOAG import read_in_trainset, read_in_predictset, random_init
from FeatureSeletion.tools import train_svm, train_knn, train_tree, num_to_feature, read_data_feature

if __name__=='__main__':
    #训练集
    trainset_file_dir = 'C:\processed_data\sonar\\'
    trainset_file_dir = trainset_file_dir.replace('\\', '/')
    trainset_file_name='train_2.txt'
    trainx,trainy=read_in_trainset(trainset_file_dir, trainset_file_name)
    #预测集
    predictset_file_dir = 'C:\processed_data\sonar\\'
    predictset_file_dir = predictset_file_dir.replace('\\', '/')
    predictset_file_name = 'predict_2.txt'
    predictx, predicty=read_in_predictset(predictset_file_dir, predictset_file_name)

    # original_acc = train_svm(trainx, trainy, predictx, predicty)
    original_acc= train_knn(trainx, trainy,predictx, predicty)
    # original_acc=train_tree(trainx, trainy,predictx, predicty)
    print original_acc


    featureList = []  # 特征集合索引,特征集合的角标
    for i in range(0,len(trainx[0])):
        featureList.append(i)

    init_forest=random_init(50,len(trainx[0]))
    for num in init_forest:
        feature=num_to_feature(num,featureList)
        train_sample=read_data_feature(feature,trainx)
        predict_sample=read_data_feature(feature,predictx)
        acc = train_knn(train_sample, trainy, predict_sample, predicty)
        print num
        print acc
        print "-------------"

