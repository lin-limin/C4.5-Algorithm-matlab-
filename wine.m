clear;
clc;
% traindata = textread('TrainData.txt');  
% testdata = textread('TestData.txt');
winedata=textread('WineData.txt');



runtime=100;
for i=1:runtime
    
%     tic;%用于计算程序运行时间
    
    train_index=randperm(length(winedata),floor(length(winedata)/4*3));%随机采样，3/4数据作为训练样本，其余的作为测试样本
    test_index=setdiff(linspace(1,length(winedata),length(winedata)),train_index);
    traindata=winedata(train_index,:);%训练样本
    testdata=winedata(test_index,:);%测试样本
    
    
    train_features=traindata(:,2:(size(traindata,2)));  
    train_targets=traindata(:,1)';  
    test_features=testdata(:,2:(size(testdata,2)));  
    test_targets=testdata(:,1)';
    
    test_targets_predict1 = C4_5(train_features', train_targets, test_features'); %调用C4.5算法用于分类
    %test_targets_predict = C4_5(train_features', train_targets, test_features', 5, 10);
    
    t=classregtree(train_features,train_targets');%调用CART算法用于分类
    test_targets_predict2=eval(t,test_features);
    
    %train_features'行是feature，列是样本  
    %train_targets 是1行多列，列是训练样本个数  
    % test_features'行是feature，列是样本 
    
    %计算决策树预测的准确度
    accuracy(i,1)=cal_accuracy(test_targets,test_targets_predict1);
    accuracy(i,2)=cal_accuracy(test_targets,test_targets_predict2');

%    t1(i)=toc;
%    save t1.mat t1;
end  


% plot(accuracy(:,1));
% hold on;
% plot(repmat(mean(accuracy(:,1)),length(accuracy(:,1)),1),'r--');
% legend('准确度','准确度均值');
% title('C4.5算法分类准确度');
% xlabel('测试次数');
% ylabel('分类准确度');
% ylim([0,1.2]);
% grid;

plot(accuracy(:,1),'--');
hold on;
plot(accuracy(:,2));
legend('C4.5 Algorithm','CART Algorithm');
xlabel('测试次数');
ylabel('分类准确度');
title('C4.5和CART分类算法的比较');
%ylim([0,1.2]);
grid;