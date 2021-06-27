# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:39:08 2019

@author: 11876
"""

from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
import time
t1=time.time()

#读入数据：
train=pd.read_csv(open('train.csv',encoding='utf-8'))
test=pd.read_csv(open('20190506_test.csv',encoding='utf-8'))

train['label']=train['label'].apply(lambda x:1 if x=='Positive' else 0)
train_y=train['label']
del(train['label'])

def stacking_first(train, train_y, test):
    savepath = './stack_op{}_dt{}_tfidf{}/'.format(args.option, args.data_type, args.tfidf)
    os.makedirs(savepath, exist_ok=True)

    count_kflod = 0
    num_folds = 6
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    # 测试集上的预测结果
    predict = np.zeros((test.shape[0], config.n_class))
    # k折交叉验证集的预测结果
    oof_predict = np.zeros((train.shape[0], config.n_class))
    scores = []
    f1s = []

    for train_index, test_index in kf.split(train):
        # 训练集划分为6折，每一折都要走一遍。那么第一个是5份的训练集索引，第二个是1份的测试集，此处为验证集是索引

        kfold_X_train = {}
        kfold_X_valid = {}

        # 取数据的标签
        y_train, y_test = train_y[train_index], train_y[test_index]
        # 取数据
        kfold_X_train, kfold_X_valid = train[train_index], train[test_index]

        # 模型的前缀
        model_prefix = savepath + 'DNN' + str(count_kflod)
        if not os.path.exists(model_prefix):
            os.mkdir(model_prefix)

        M = 4  # number of snapshots
        alpha_zero = 1e-3  # initial learning rate
        snap_epoch = 16
        snapshot = SnapshotCallbackBuilder(snap_epoch, M, alpha_zero)

        # 使用训练集的size设定维度，fit一个模型出来
        res_model = get_model(train)
        res_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # res_model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,  class_weight=class_weight)
        res_model.fit(kfold_X_train, y_train, batch_size=BATCH_SIZE, epochs=snap_epoch, verbose=1,
                      validation_data=(kfold_X_valid, y_test),
                      callbacks=snapshot.get_callbacks(model_save_place=model_prefix))

        # 找到这个目录下所有已经训练好的深度学习模型，通过".h5"
        evaluations = []
        for i in os.listdir(model_prefix):
            if '.h5' in i:
                evaluations.append(i)

        # 给测试集和当前的验证集开辟空间，就是当前折的数据预测结果构建出这么多的数据集[数据个数，类别]
        preds1 = np.zeros((test.shape[0], config.n_class))
        preds2 = np.zeros((len(kfold_X_valid), config.n_class))
        # 遍历每一个模型，用他们分别预测当前折数的验证集和测试集，N个模型的结果求平均
        for run, i in enumerate(evaluations):
            res_model.load_weights(os.path.join(model_prefix, i))
            preds1 += res_model.predict(test, verbose=1) / len(evaluations)
            preds2 += res_model.predict(kfold_X_valid, batch_size=128) / len(evaluations)

        # 测试集上预测结果的加权平均
        predict += preds1 / num_folds
        # 每一折的预测结果放到对应折上的测试集中，用来最后构建训练集
        oof_predict[test_index] = preds2

        # 计算精度和F1
        accuracy = mb.cal_acc(oof_predict[test_index], np.argmax(y_test, axis=1))
        f1 = mb.cal_f_alpha(oof_predict[test_index], np.argmax(y_test, axis=1), n_out=config.n_class)
        print('the kflod cv is : ', str(accuracy))
        print('the kflod f1 is : ', str(f1))
        count_kflod += 1

        # 模型融合的预测结果，存起来，用以以后求平均值
        scores.append(accuracy)
        f1s.append(f1)
    # 指标均值，最为最后的预测结果
    print('total scores is ', np.mean(scores))
    print('total f1 is ', np.mean(f1s))
    return predict
predict=stacking_first(train, train_y, test)