[Num_1]=xlsread('ML_data2_trans',1);
[Num_2]=xlsread('ML_data2_trans',2);
train_data=Num_1(:,1:14);
train_label=Num_1(:,15);
test_data=Num_2(:,1:14);
test_label=Num_2(:,15);

%KNNÀ„∑®
mdl = ClassificationKNN.fit(train_data,train_label,'NumNeighbors',1);
knn_predict_label= predict(mdl, test_data);
knn_accuracy     = length(find(knn_predict_label == test_label))/length(test_label)*100;

%RandomF
nTree = 80;
B = TreeBagger(nTree,train_data,train_label');
RandomF_predict_label = predict(B,test_data);
RandomF_predict_label = str2double(RandomF_predict_label);
RandomF_accuracy = length(find(RandomF_predict_label == test_label))/length(test_label)*100;

%Naive Bayes
nb = fitcnb(train_data, train_label);
NB_predict_label = predict(nb, test_data);
NB_accuracy = length(find(NB_predict_label == test_label))/length(test_label)*100;

%AdaBoost
ens = fitensemble(train_data,train_label,'AdaBoostM1',100,'tree','type','classification');
AB_predict_label = predict(ens, test_data);
AB_accuracy = length(find(AB_predict_label == test_label))/length(test_label)*100;

%Discriminant Analysis Classifier
obj = ClassificationDiscriminant.fit(train_data, train_label);
DAC_predict_label = predict(obj, test_data);
DAC_accuracy = length(find(DAC_predict_label == test_label))/length(test_label)*100;

% %SVM
% option = statset('MaxIter',30000);
% svmModel = svmtrain(train_data, train_label,'kernel_function','rbf',option);
% svm_predict_label = svmclassify(svmModel,test_data); 
% svm_accuracy =sum(svm_predict_label==test_label)/size(test_label,1);





