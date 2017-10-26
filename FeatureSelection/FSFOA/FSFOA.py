#coding=utf-8
from numpy import *
from tools import *
import itertools#处理list的嵌套问题
#fea_combination=2**num_fea_original#特征排列组合后的个数
#print('原数据集特征数目：',num_fea_original)

loop_condition=8#最起码要大于lifetime=15的值 ，因为播种一次age才曾1
initialization_parameters = [15, 12, 30, 0.05, 50]

file_dir='C:\processed_data\sonar\\'
file_dir=file_dir.replace('\\','/')

trainX,trainy=load_data(file_dir+'train_2.txt')#trainX,trainy are all list
predictX,predicty=load_data(file_dir+'predict_2.txt')

num_tree_ini=60#初始化时森林中tree的个数
num_fea_original=mat(trainX).shape[1]
feature=[]#特征集合索引,特征集合的角标
for i in range(num_fea_original):
    feature.append(i)



original_acc= train_knn(trainX, trainy,predictX, predicty)
#original_acc=train_svm(trainX, trainy,predictX, predicty)
#original_acc=train_tree(trainX, trainy,predictX, predicty)

def reverse_binary_GSC(vice_verse_attri_GSC,candidate_area,num_fea_original):
    after_reverse=[]
    selected_tree_canarea=[]#从候选区中挑出来进行反转的树
    num_percent_transfer=int(len(candidate_area)*initialization_parameters[3])

    # 从不断增长的候选区中挑出来进行反转的树
    j=0
    x=[]#做测试用 ，可以删除
    while j <num_percent_transfer:
        y = random.randint(0, len(candidate_area) - 1)
        if candidate_area[y] not in selected_tree_canarea:
            selected_tree_canarea.append(candidate_area[y])
            j = j + 1
            x.append(y)
        else:
            continue

    for i in range(len(selected_tree_canarea)):
        temp = Tree(selected_tree_canarea[i].list, selected_tree_canarea[i].age)
        for j in range(len(vice_verse_attri_GSC)):
            if temp.list[vice_verse_attri_GSC[j]]=='0':
                const_value=1
                new_string=index_replace(vice_verse_attri_GSC[j],temp.list,const_value)
                temp.list=new_string
            else:
                const_value=0
                new_string=index_replace(vice_verse_attri_GSC[j],temp.list,const_value)
                temp.list=new_string
        after_reverse.append(temp)

    return after_reverse
#local_seeding
def reverse_binary(vice_verse_attri,area_limit_forest):#area_limit_forest_age0[i]
    after_reverse=[]
    area_limit_forest_age0 = []
    for i in range(len(area_limit_forest)):
        if area_limit_forest[i].age == 0:
            area_limit_forest_age0.append(area_limit_forest[i])  # 确保原始树送入reverse_binary反转产生新树后，原始树的age值加1。
        else:
            continue
    for i in range(len(area_limit_forest)):
        area_limit_forest[i].age += 1
    for i in range(len(area_limit_forest_age0)):
        for k in range(len(vice_verse_attri)):
            temp=Tree(area_limit_forest_age0[i].list,0)
            if temp.list[vice_verse_attri[k]]=='0':
                const_value=1
                new_string=index_replace(vice_verse_attri[k],temp.list,const_value)
                temp.list=new_string
            else:
                const_value=0
                new_string=index_replace(vice_verse_attri[k],temp.list,const_value)
                temp.list=new_string
            after_reverse.append(temp)
    return after_reverse
#population_limiting
def select_trees(area_limit_forest):
    selected_trees=[]
    acc = []
    acc_omit_index=[]#存的是acc中前num_extra的最小值的角标
    age_exceed_lifetime_index=[]#age值超过lifetime的索引号
    if len(area_limit_forest)<=initialization_parameters[4]:
        for i in range(len(area_limit_forest)):
            if area_limit_forest[i].age>initialization_parameters[0]:
               selected_trees.append(area_limit_forest[i])
               age_exceed_lifetime_index.append(i)
        delete_together(age_exceed_lifetime_index, area_limit_forest)
    else:
        for i in range(len(area_limit_forest)):
            if area_limit_forest[i].age>initialization_parameters[0]:
               selected_trees.append(area_limit_forest[i])
               age_exceed_lifetime_index.append(i)
        delete_together(age_exceed_lifetime_index, area_limit_forest)
        if len(area_limit_forest)>initialization_parameters[4]:
            #遍历area_limit_forest中剩下的树带入求解器（eg knn）算分类准确率，准确率低的放入候选区直至area_limit_forest的长度为are_limit的值为止
            num_extra=len(area_limit_forest)-initialization_parameters[4]
            print('num_extra',num_extra)
            for i in range(len(area_limit_forest)):
                fea_list=num_to_feature(area_limit_forest[i].list,feature)
                if len(fea_list):
                    data_sample=read_data_feature(fea_list,trainX)
                    data_predict=read_data_feature(fea_list,predictX)
                    acc.append(train_knn(data_sample, trainy, data_predict, predicty))#每棵树的准确率存在acc中
                    #acc.append(train_svm(trainX, trainy, predictX, predicty))
                    #acc.append(train_tree(trainX, trainy, predictX, predicty))
                else:
                    acc.append(0)
                    #exit(1)

             #将acc中前num_extra的最小值的角标存入acc_omit_index中
            for i in range(num_extra):
                acc_min = min(acc)
                acc_min_index = acc.index(acc_min)
                acc[acc_min_index] = 100
                acc_omit_index.append(acc_min_index)
            for each_item in acc_omit_index:
                selected_trees.append(area_limit_forest[each_item])
            delete_together(acc_omit_index, area_limit_forest)
    return selected_trees




class Tree:
    def __init__(self, tree_list, tree_age):
        self.age = tree_age
        self.list = tree_list


initial_forest=[]
initial_forest_index=[]
area_limit_forest=[]

#将森林中的树以字符串的形式初始化为全0/与随机产生随几个数的效果差不多
ini_str=''
for i in range(num_fea_original):
    ini_str+='0'
for i in range(num_tree_ini):
    initial_forest.append(ini_str)
#将森林初始化为list为全0的字符串，age为0
for each_item in initial_forest:
    instance=Tree(each_item,0)
    area_limit_forest.append(instance)





accuracy_max=[]#存储循环20次中每次的最大准确率
accuracy_max_feature=[]#存储循环20次中每次的最大准确率所对应的特征


candidate_area=[]
candidate_area_growing=[]
candidate_area_temp=[]
m=0
while (m<loop_condition):
    print'￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥第',m+1,'次循环￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥'
    m += 1
    vice_verse_attri=[]
    j=0
    while j<initialization_parameters[1]:
        y = random.randint(0, num_fea_original - 1)
        if y not in vice_verse_attri:
            vice_verse_attri.append(y)
            j=j+1
        else:
            continue
    print'局部播种时选出的需要反转属性',vice_verse_attri
    print'增加新树前area_limit_forest的长度', len(area_limit_forest)



    # print'######################################第',m,'local seeding播种开始######################################'
    new_tree_nestification=[]#reverse_binary函数调用后返回的list会有嵌套
    new_tree_nonestification=[]
    new_tree_nestification.append(reverse_binary(vice_verse_attri,area_limit_forest))
    new_tree_nonestification=list(itertools.chain.from_iterable(new_tree_nestification))
    # print'new_tree_nonestification增长出新树的长度', len(new_tree_nonestification)
    # for i in range(len(new_tree_nonestification)):
    #     print'新生成的树',new_tree_nonestification[i].list,new_tree_nonestification[i].age
    # for i in range(len(area_limit_forest)):
    #     print'插入新树之前area_limit_forest的样子',area_limit_forest[i].list,area_limit_forest[i].age
#向area_limit里插入新树
    for each_item in new_tree_nonestification:
        area_limit_forest.append(each_item)
    # print'加入新树后area_limit_forest的长度',len(area_limit_forest)
    # for i in range(len(area_limit_forest)):
    #     print'新生成的area_limit_forest',area_limit_forest[i].list,area_limit_forest[i].age

    # print'######################################第',m,'local seeding播种开始结束######################################'


    # print'######################################第',m,'population limiting 放入候选区开始##########################################'
    candidate_area_growing.append(select_trees(area_limit_forest))
    #new_tree_nonestification = list(itertools.chain.from_iterable(new_tree_nestification))
    candidate_area_temp=list(itertools.chain.from_iterable( candidate_area_growing))
    # for i in range(len(candidate_area_temp)):
    #     if isinstance(candidate_area_temp[i].list,str):
    #         candidate_area_temp[i].list=int(candidate_area_temp[i].list,2)
    #     else:
    #         continue
    candidate_area=candidate_area_temp
    # print'候选区candidate_area的长度：',len(candidate_area)
    # for i in range(len(candidate_area)):
    #     print'候选区candidate_area中准确率最小前num_extra颗树的list值，age值：',candidate_area[i].list,candidate_area[i].age
    # print'移除多余树area_limit_forest的长度',len(area_limit_forest)
    # for i in range(len(area_limit_forest)):
    #     print'移除多余树area_limit_forest的list值，age值：',area_limit_forest[i].list,area_limit_forest[i].age
    # print'#####################################第',m,'population limiting放入候选区结束##########################################'
    #
    #
    # print'######################################第',m,'Global seeding GSC开始##########################################'
#只需要根据GSC值完成候选区5%的反转即可
    vice_verse_attri_GSC=[]
    j=0
    while j<initialization_parameters[2]:
        y = random.randint(0, num_fea_original - 1)
        if y not in vice_verse_attri_GSC:
            vice_verse_attri_GSC.append(y)
            j=j+1
        else:
            continue
    after_GSC_reverse=[]#存放经过reverse_binary_GSC反转后的结果
    after_GSC_reverse=reverse_binary_GSC(vice_verse_attri_GSC,candidate_area,num_fea_original)
    area_limit_forest+=after_GSC_reverse

    acc=[]
    for i in range(len(area_limit_forest)):
        fea_list = num_to_feature(area_limit_forest[i].list, feature)
        if len(fea_list):
            data_sample = read_data_feature(fea_list, trainX)
            data_predict = read_data_feature(fea_list, predictX)
            acc.append(train_knn(data_sample, trainy, data_predict, predicty))# 每棵树的准确率存在acc中
            #acc.append(train_svm(trainX, trainy, predictX, predicty))
            #acc.append(train_tree(trainX, trainy, predictX, predicty))
        else:
            acc.append(0)
    # print('acc准确率的长度',len(acc))
    # print('acc里面的准确率值',acc)
    acc_max=max(acc)
    print acc
    accuracy_max.append(acc_max)
    #print('最大准确率',acc_max)
    acc_max_index=acc.index(acc_max)
    #print('最大准确率索引值',acc_max_index)
    tree_max=area_limit_forest[acc_max_index]#找到最优树在area_limit_forest中的位置,最优树即准确率最高的特征子集。
    accuracy_max_feature.append(tree_max.list)
    #print('最优树即准确率最高的特征子集:',tree_max.list,tree_max.age)
    area_limit_forest[acc_max_index].age=0#最优树的age设为0

    print'#####################################第',m,'update optimal更新最优结束##########################################'


print'feature index : ', feature
print'feature number of Original data set :  ',num_fea_original#原始数据集特征数目
print'original_accuracy is : ',original_acc
print'FSFOA_accuracy is : ',float(max(accuracy_max))
print'Feature subset ：',accuracy_max_feature[accuracy_max.index(max(accuracy_max))]
print'length of candidate area ：',len(candidate_area)
#处理dimension reduction
num_one_index=num_to_feature(accuracy_max_feature[accuracy_max.index(max(accuracy_max))],feature)#被选中的最优特征子集中‘1’的索引，以列表的形式返回
num_one=len(num_one_index)
DR=float(1-(1.0*num_one/len(feature)))
print'percent of dimension reduction : ',DR
