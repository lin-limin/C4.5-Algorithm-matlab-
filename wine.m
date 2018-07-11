clear;
clc;
% traindata = textread('TrainData.txt');  
% testdata = textread('TestData.txt');
winedata=textread('WineData.txt');



runtime=100;
for i=1:runtime
    
%     tic;%���ڼ����������ʱ��
    
    train_index=randperm(length(winedata),floor(length(winedata)/4*3));%���������3/4������Ϊѵ���������������Ϊ��������
    test_index=setdiff(linspace(1,length(winedata),length(winedata)),train_index);
    traindata=winedata(train_index,:);%ѵ������
    testdata=winedata(test_index,:);%��������
    
    
    train_features=traindata(:,2:(size(traindata,2)));  
    train_targets=traindata(:,1)';  
    test_features=testdata(:,2:(size(testdata,2)));  
    test_targets=testdata(:,1)';
    
    test_targets_predict1 = C4_5(train_features', train_targets, test_features'); %����C4.5�㷨���ڷ���
    %test_targets_predict = C4_5(train_features', train_targets, test_features', 5, 10);
    
    t=classregtree(train_features,train_targets');%����CART�㷨���ڷ���
    test_targets_predict2=eval(t,test_features);
    
    %train_features'����feature����������  
    %train_targets ��1�ж��У�����ѵ����������  
    % test_features'����feature���������� 
    
    %���������Ԥ���׼ȷ��
    accuracy(i,1)=cal_accuracy(test_targets,test_targets_predict1);
    accuracy(i,2)=cal_accuracy(test_targets,test_targets_predict2');

%    t1(i)=toc;
%    save t1.mat t1;
end  


% plot(accuracy(:,1));
% hold on;
% plot(repmat(mean(accuracy(:,1)),length(accuracy(:,1)),1),'r--');
% legend('׼ȷ��','׼ȷ�Ⱦ�ֵ');
% title('C4.5�㷨����׼ȷ��');
% xlabel('���Դ���');
% ylabel('����׼ȷ��');
% ylim([0,1.2]);
% grid;

plot(accuracy(:,1),'--');
hold on;
plot(accuracy(:,2));
legend('C4.5 Algorithm','CART Algorithm');
xlabel('���Դ���');
ylabel('����׼ȷ��');
title('C4.5��CART�����㷨�ıȽ�');
%ylim([0,1.2]);
grid;