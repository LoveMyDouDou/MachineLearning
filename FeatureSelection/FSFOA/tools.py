#coding=utf-8
from numpy import mat
from sklearn import neighbors, svm, tree


def load_data(filename):
    '''
    读入特征选择文件，文件每行以逗号分隔
    :param filename: 文件名称
    :return: feature特征列表，是一个二维列表，存储特征矩阵
             label 标签列表，一维列表，存储每条记录对应的分类
    '''
    numOfFeature=len(open(filename).readline().split(','))-1
    feature=[]
    label=[]
    fr=open(filename)
    for line in fr.readlines():
        xi=[]
        currentLine=line.strip().split(',')# curline=line.strip().split('\t')
        for i in range(0,numOfFeature):
            xi.append(float(currentLine[i]))
        feature.append(xi)
        label.append(float(currentLine[-1]))
    return feature,label


def read_data_feature(fetureList,dataSet):
    dataMat=mat(dataSet)
    row=dataMat.shape[0]#行数
    data_sample=[]
    for i in range(0,row):
        row_i=[]
        for j in fetureList:
            row_i.append(dataMat[i,j])
        data_sample.append(row_i)
    return data_sample


def index_replace(index,replaceString,constValue):
    '''
    在指定角标索引的位置将字符串进行替换
    :param index: 替换字符的角标值
    :param replaceString: 被替换的字符串
    :param constValue: 替换的新字符
    :return: 在给定位置替换新字符后的新字符串
    '''
    newString=''
    for i in range(len(replaceString)):
        if i!=index:
            newString+=replaceString[i]
        else:
            newString+=str(constValue)
    return newString


def delete_together(delete_index,a):
    '''
    给定索引列表delete_index，统一删除列表a中相应索引的元素
    :param delete_index:删除元素的下标
    :param a: 被删除元素的列表
    :return: 无返回值，注意列表a被删除
    '''
    for i in delete_index:
        a[i]='k'
    for i in range(len(delete_index)):
        a.remove('k')


def acc_pre(label_pre,label_train):
    '''
    预测标签和ground_true标签对比 算准确率
    :param label_pre: 预测标签列表
    :param label_train: 实际标签列表
    :return: 预测准确率
    '''
    num=0
    for i in range(len(label_pre)):
        if label_pre[i]==label_train[i]:
            num+=1
    return 1.0*num/len(label_train)


def train_knn(data_train,label_train,data_pre,label_pre):
    '''
    KNN分类器
    :param data_train: 训练数据集合
    :param label_train: 训练标签集合
    :param data_pre: 预测数据集合
    :param label_pre: 预测标签集合
    :return: KNN分类器下的准确率
    '''
    clf=neighbors.KNeighborsClassifier(n_neighbors=1)#创建分类器对象
    clf.fit(data_train,label_train)#用训练数据拟合分类器模型搜索
    predict=clf.predict(data_pre)
    acc=acc_pre(predict,label_pre)# 预测标签和ground_true标签对比 算准确率
    return acc


def train_svm(data_train,label_train,data_predict,label_predict):
    '''
    支持向量机分类器
    :param data_train: 训练数据集合
    :param label_train: 训练标签集合
    :param data_predict: 预测数据集合
    :param label_predict: 预测标签集合
    :return: SVM分类器下的准确率
    '''
    clf=svm.SVC()
    clf.fit(data_train,label_train)
    predict=clf.predict(data_predict)
    acc=acc_pre(predict,label_predict)
    return acc


def train_tree(data_train,label_train,data_pre,label_pre):
    '''
    决策树分类器
    :param data_train: 训练数据集合
    :param label_train: 训练标签集合
    :param data_pre: 预测数据集合
    :param label_pre: 预测标签集合
    :return: SVM分类器下的准确率
    '''
    #dot_data=StringIO()
    clf=tree.DecisionTreeClassifier()
    clf.fit(data_train,label_train)
    # tree.export_graphviz(clf,out_file=dot_data)
    # graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf('wine.pdf')
    predict=clf.predict(data_pre)
    acc=acc_pre(predict,label_pre)
    return acc


def num_to_feature(num,feature_list):
    '''
    将一串01数字转化为特征向量表，位置上为'1'的构造成新的向量
    :param num: 一个字符串，字符串中只包含01
    :param feature_list: 原始的特征向量
    :return: 产生新的特征向量
    '''
    feature=[]
    for i in range(len(num)):
        if num[i]=='1':
            feature.append(feature_list[i])
    return feature









