# -*- coding: utf-8 -*-
import csv
from random import seed
from random import randrange


'''
读取csv数据(代码存在不足之处:运行时间较长，演示的话尽量采用第二种读入少量数据方法)
'''
def loadCSV(filename): 
    dataSet=[]
    with open(filename,'r') as file:
        csvReader=csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet

''' #仅提取部分数据集中的元素进行决策树训练的方法(用于减少在测试代码过程中所花费的时间)
def loadCSV(filename):
    dataSet=[]
    with open(filename,'r') as file:
        csvReader=csv.reader(file)
        for i,line in enumerate(csvReader):
            if i < 1000:
                dataSet.append(line)
    return dataSet
'''

'''
除了标签列其余的数据全部转换为float类型
'''
def column_to_float(dataSet):
    feature_len=len(dataSet[0])-1
    for data in dataSet:
        for column in range(feature_len):
            #.strip()用于移除字符串首尾指定的字符
            data[column]=float(data[column].strip()) #强制转换为float数据类型

'''
将数据集分成N块，方便进行交叉验证,这里是把数据集平均分成了N份，每一份不包含相同的元素
'''
def splitDataSet(dataSet,n_folds):
    fold_size=int(len(dataSet)/n_folds)
    #print (fold_size)        #用于检测代码正确性，记得删掉
    dataSet_copy=list(dataSet)
    dataSet_spilt=[]
    for i in range(n_folds):
        fold=[]
        while len(fold) < fold_size:   #while执行循环，直到条件不成立
            index=randrange(len(dataSet_copy))
            fold.append(dataSet_copy.pop(index))  #pop()函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
        dataSet_spilt.append(fold)
    return dataSet_spilt

'''
构造数据子集
'''
def get_subsample(dataSet,ratio):
    subdataSet=[]
    lenSubdata=round(len(dataSet)*ratio)
    #print (lenSubdata)        #用于检测代码正确性，记得删掉
    while len(subdataSet) < lenSubdata:
        index=randrange(len(dataSet)-1)
        subdataSet.append(dataSet[index])
    return subdataSet

'''
分割数据集(根据某一特征值的值作为阈值将数据集分割为两部分，作为树的左右分支)
'''
def data_split(dataSet,index,value):
    data_left=[]
    data_right=[]
    for data_row in dataSet:
        if data_row[index]<value:
            data_left.append(data_row)
        else:
            data_right.append(data_row)
    return data_left,data_right

'''
计算分割代价,也就是基尼指数(对于二分类问题，计算出的分割代价是P1(1-P1)+P2(1-P2)+P3(1-P3)+P4(1-P4))
其中，P1是分割的左侧数据集中类别为0的样本比例，同理，P2是类别1的样本比例。用于选取最优分割特征
'''
def spilt_loss(left,right,class_values):
    loss=0.0
    for class_value in class_values:
        left_size=len(left)
        if left_size!=0:  #防止除数为零
            #.count()方法用于统计字符串里某个字符出现的次数,这里是统计左列表中分类为class_value的样本有多少个。
            prop=[row[-1] for row in left].count(class_value)/float(left_size)
            loss += (prop*(1.0-prop)) #prop是被分成的左侧列表中类别为class_value的样本所占的比例。
        right_size=len(right)
        if right_size!=0:
            prop=[row[-1] for row in right].count(class_value)/float(right_size)
            loss += (prop*(1.0-prop))
    return loss

'''
选取任意的n个特征和包含的数据集，通过判断分割代价找到最优的分割特征，其中n_features为选取分割特征的个数
'''
def get_best_spilt(dataSet,n_features):
    features=[]
    class_values=list(set(row[-1] for row in dataSet)) #即class_values返回值为此时数据集中包含的全部标签组成的列表
    b_index,b_value,b_loss,b_left,b_right=9999,9999,9999,None,None #初始化最佳分割特征索引，最佳分割阈值，最佳分割时的分割代价，以及最佳分割以后产生的左、右子数据集
    while len(features) < n_features:
        index = randrange(len(dataSet[0])-1)
        if index not in features:   #注意python中关于列表操作做的这种表达方式
            features.append(index)
    for index in features:   #注意：对于全部的特征，都以数据集中全部样本关于此特征对应的值作为阈值进行分割，求解最优分割方式，这样做会导致运行非常缓慢。
        for value_set in set(row[index] for row in dataSet): 
            left,right=data_split(dataSet,index,value_set)
            loss=spilt_loss(left,right,class_values)
            if  loss < b_loss:
                b_index,b_value,b_loss,b_left,b_right = index,value_set,loss,left,right
    return {'index':b_index,'value':b_value,'left':b_left,'right':b_right}  
    '''注意返回的字典格式。'''  
    
'''
决定输出标签,即对于一个叶节点的数据子集，选取子集中包含最多的标签作为整个节点的类标签
'''
def decide_label(data):
    output=[row[-1] for row in data]
    return max(set(output),key=output.count)
    #利用key回调函数判断max的语句，整体可以理解为根据key来判断前述list中的最大值，这里就是找出出现次数最多的label
    
'''
子分割，不断构建叶节点的过程。
root:此时分割到的节点;  max_depth:定义的决策树的最大深度；depth:此时节点root所在的深度
min_size:定义的继续可分的中间节点最少应该包含的样本数量;
'''
def sub_spilt(root,n_features,max_depth,min_size,depth):   
    left=root['left']
    right=root['right']
    del(root['left'])
    del(root['right'])
    if not left or not right:  #缺少左子树与右子树的处理
        root['left']=root['right']=decide_label(left+right)
        return
    if depth > max_depth:  #树深不能超过定义的最大深度（防止过拟合）
        root['left']=decide_label(left)
        root['right']=decide_label(right)
        return
    if len(left) < min_size: #包含样本量较少时直接作为叶子节点
        root['left']=decide_label(left)
    else:  #持续进行分割过程
        root['left'] = get_best_spilt(left,n_features)
        sub_spilt(root['left'],n_features,max_depth,min_size,depth+1)
    if len(right) < min_size:  
        root['right']=decide_label(right)
    else:
        root['right'] = get_best_spilt(right,n_features)
        sub_spilt(root['right'],n_features,max_depth,min_size,depth+1)

'''
构造决策树
'''
def build_tree(dataSet,n_features,max_depth,min_size):
    root=get_best_spilt(dataSet,n_features)
    sub_spilt(root,n_features,max_depth,min_size,1) 
    return root

'''
对于一个样本row,预测其被某一棵树tree分类得到的结果
'''
def predict(tree,row):
    #predictions=[]
    if row[tree['index']] < tree['value']: #错误原因：row[]而不是row(),提示错误：'list' object is not callable
        if isinstance(tree['left'],dict): #isinstance用来判断一个对象是否是一个已知的类型,此处判断此节点的是否还有左子树继续分类。
            return predict(tree['left'],row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'],dict):
            return predict(tree['right'],row)
        else:
            return tree['right']
   # predictions=set(predictions)



'''
对于一个样本row,预测其被森林(多棵树)分类得到的结果
'''           
def bagging_predict(trees,row):
    predictions=[predict(tree,row) for tree in trees]
    return max(set(predictions),key=predictions.count)#返回各决策树中的最多表决结果
    
'''
创建随机森林
'''
def random_forest(train,test,ratio,n_features,max_depth,min_size,n_trees):
    trees=[]
    for i in range(n_trees): #n_trees表示决策树的数量
        train_change=get_subsample(train,ratio) #随机采样保证了每棵决策树训练集的差异性
        tree=build_tree(train_change,n_features,max_depth,min_size) #建立一个决策树
        trees.append(tree)
    predict_values = [bagging_predict(trees, row) for row in test]
    return predict_values #返回全部测试集的预测标签

'''
计算预测得到的准确率
'''
def accuracy(predict_values,actual):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predict_values[i]:
           correct += 1
    return correct/float(len(actual))

'''
主函数
'''
if __name__=='__main__':
    seed(1)
    train_dataSet=loadCSV('ML_data2_train.csv')
    test_dataSet=loadCSV('ML_data2_test.csv')
    column_to_float(train_dataSet)
    column_to_float(test_dataSet)
    max_depth=30  #每棵树的最大发展深度
    min_size=1 #每个能继续可分的节点最少应该包含的样本数量
    ratio=0.8 #每次训练构造数据子集时数据子集中包含的样本数占总数据集中样本数的比例
    n_features=13 #每次构造决策树时从总特征数中随机选择的特征个数
    n_trees=80 #随机森林中决策树的数量
    train_Set=train_dataSet #训练集
    test_Set=[] #测试集
    for row in test_dataSet:
        row_copy=list(row)
        row_copy.pop(14)
        test_Set.append(row_copy) #测试集的类别信息
    actual=[row[-1] for row in test_dataSet] #测试集的真实结果   
    predict_values=random_forest(train_Set,test_Set,ratio,n_features,max_depth,min_size,n_trees)
    accur=accuracy(predict_values,actual)
    print ('accur:%s'% accur)
  
'''       
    for n_trees in range(1,101):
        if n_trees%5==0:
            predict_values=random_forest(train_Set,test_Set,ratio,n_features,max_depth,min_size,n_trees)
            accur=accuracy(predict_values,actual)
            print ('tree_num:%s,accur:%s'% (n_trees,accur))
'''     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




            