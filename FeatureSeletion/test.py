#coding=utf-8
from FeatureSeletion.FSFOAG import read_in_trainset, read_in_predictset, random_init, one_point_hybridization, \
    one_point_hybridization_knn, one_point_hybridization_knn_result
from FeatureSeletion.tools import train_svm, train_knn, train_tree, num_to_feature, read_data_feature, num_to_string, \
    string_to_numlist, num_to_list, calculate_DR

if __name__=='__main__':
    #训练集
    # trainset_file_dir = 'C:\processed_data\\'
    trainset_file_dir = 'C:\processed_data\glass\\'
    # trainset_file_dir = 'C:\processed_data\sonar\\'
    # trainset_file_dir = 'C:\processed_data\SRBCT\\'
    # trainset_file_dir = trainset_file_dir.replace('\\', '/')
    trainset_file_name='train_702.txt'
    trainx,trainy=read_in_trainset(trainset_file_dir, trainset_file_name)
    #预测集
    # predictset_file_dir = 'C:\processed_data\\'
    predictset_file_dir = 'C:\processed_data\glass\\'
    # predictset_file_dir = 'C:\processed_data\sonar\\'
    # predictset_file_dir = 'C:\processed_data\SRBCT\\'
    predictset_file_dir = predictset_file_dir.replace('\\', '/')
    predictset_file_name = 'predict_302.txt'
    predictx, predicty=read_in_predictset(predictset_file_dir, predictset_file_name)

    # original_acc = train_svm(trainx, trainy, predictx, predicty)
    # original_acc= train_knn(trainx, trainy,predictx, predicty,1)
    # original_acc=train_tree(trainx, trainy,predictx, predicty)
    # print original_acc


    feature_list = []  # 特征集合索引,特征集合的角标
    for i in range(0,len(trainx[0])):
        feature_list.append(i)

    forest={}  #记录森林里的准确率

    init_forest=random_init(50,len(trainx[0]))
    # for i in init_forest:
    #     print i

    # print 'trainx',trainx
    # print 'trainy',trainy

    for num in init_forest:
        feature=num_to_feature(num,feature_list)
        train_sample=read_data_feature(feature,trainx)
        predict_sample=read_data_feature(feature,predictx)
        # print 'train_sample',train_sample
        # print 'train_y',trainy
        # print 'predict_sample',predict_sample
        # print 'predict_y',predicty
        acc = train_knn(train_sample, trainy, predict_sample, predicty,1)

        num_string=num_to_string(num)
        forest[num_string]=acc


    # forest_area = []
    forest_old=init_forest
    res={}
    for i in range(0,5):
        forest,forest_new=one_point_hybridization_knn(forest_old,feature_list,trainx,trainy,predictx,predicty,1)
        forest_list=sorted(forest.items(),key=lambda item:item[1],reverse=True)
        forest_next=[]
        for j in range(0,10):
            forest_next.append(forest_old[j])
        for j in range(0,40):
            forest_next.append(num_to_list(forest_list[j][0]))
        forest_old=forest_next
        res[forest_list[0][0]]=forest_list[0][1]

        # for j in forest_list:
        #     print j
        # forest_area.append(string_to_numlist(forest_list[0][0]))
    ans=sorted(res.items(),key=lambda item:item[1],reverse=True)
    for i in ans:
        print 'accuracy=',i[1],'DR=',calculate_DR(i[0])
