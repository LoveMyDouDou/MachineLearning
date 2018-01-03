#coding=utf-8
from FeatureSeletion import read_in_predictset, random_init, one_point_hybridization_knn
from FeatureSeletion.FSFOAG import one_point_hybridization_svm, one_point_hybridization_train_tree
from FeatureSeletion.tools import num_to_list, calculate_DR

def DO_FSFOA_HIGH(file_dir_name, predictset_file_dir, predictset_file_name, trainx, trainy, loop):
    predictx, predicty = read_in_predictset(predictset_file_dir, predictset_file_name)

    feature_list = []  # 特征集合索引,特征集合的角标
    for i in range(0, len(trainx[0])):
        feature_list.append(i)
    forest = {}  # 记录森林里的准确率
    init_forest = random_init(50, len(trainx[0]))

    write_to = file_dir_name + '_' + predictset_file_name + '_result.txt'
    file_object = open(write_to, 'a')
    file_object.write(predictset_file_name + ':\n')

    for neighbor in range(1, 6, 2):
        tree, acc = one_point_hybridization_knn_high_result(init_forest, feature_list, trainx, trainy, predictx,
                                                       predicty, loop, neighbor)
        file_object.write('knn n=')
        file_object.write(str(neighbor))
        file_object.write('\n')
        file_object.write('accuracy=')
        file_object.write(str(acc))
        file_object.write('\n')
        file_object.write('DR=')
        file_object.write(str(calculate_DR(tree)))
        file_object.write('\n')

    tree, acc = one_point_hybridization_svm_high_result(init_forest, feature_list, trainx, trainy, predictx, predicty,
                                                   loop)
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

    tree, acc = one_point_hybridization_train_tree_high_result(init_forest, feature_list, trainx, trainy, predictx,
                                                          predicty, loop)
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


def one_point_hybridization_knn_high_result(forest_old, feature_list, trainx, trainy, predictx, predicty,loop,neighbor):
    res = {}
    for i in range(0, loop):
        forest, forest_new = one_point_hybridization_knn(forest_old, feature_list, trainx, trainy, predictx, predicty,neighbor)
        forest_list = sorted(forest.items(), key=lambda item:(item[1],item[0]), reverse=True)
        forest_next = []
        for j in range(0, 10):
            forest_next.append(forest_old[j])
        for j in range(0, 40):
            forest_next.append(num_to_list(forest_list[j][0]))
        forest_old = forest_next
        # res.append(forest_list[0][1])
        res[forest_list[0][0]]=forest_list[0][1]
    first=sorted(res.items(), key=lambda item:(item[1],item[0]),reverse=True)[0]
    return first


def one_point_hybridization_svm_high_result(forest_old, feature_list, trainx, trainy, predictx, predicty,loop):
    res = {}
    for i in range(0, loop):
        forest, forest_new = one_point_hybridization_svm(forest_old, feature_list, trainx, trainy, predictx, predicty)
        forest_list = sorted(forest.items(), key=lambda item:(item[1],item[0]), reverse=True)
        forest_next = []
        for j in range(0, 10):
            forest_next.append(forest_old[j])
        for j in range(0, 40):
            # print 'forest_list.length=',len(forest_list)
            # print 'j=',j
            forest_next.append(num_to_list(forest_list[j][0]))
        forest_old = forest_next
        # res.append(forest_list[0][1])
        res[forest_list[0][0]]=forest_list[0][1]
    first=sorted(res.items(), key=lambda item:(item[1],item[0]),reverse=True)[0]
    return first


def one_point_hybridization_train_tree_high_result(forest_old, feature_list, trainx, trainy, predictx, predicty,loop):
    res = {}
    for i in range(0, loop):
        forest, forest_new = one_point_hybridization_train_tree(forest_old, feature_list, trainx, trainy, predictx, predicty)
        forest_list = sorted(forest.items(), key=lambda item:(item[1],item[0]), reverse=True)
        forest_next = []
        for j in range(0, 10):
            forest_next.append(forest_old[j])
        for j in range(0, 40):
            forest_next.append(num_to_list(forest_list[j][0]))
        forest_old = forest_next
        # res.append(forest_list[0][1])
        res[forest_list[0][0]]=forest_list[0][1]
    first=sorted(res.items(), key=lambda item:(item[1],item[0]),reverse=True)[0]
    return first

