function test_targets = improved_C4_5(train_features, train_targets, test_features,varargin)    
    
pruning=35;
thres_disc=10;
if nargin>4
    pruning=varargin{1};
    thres_disc=varargin{2};
elseif nargin>3
    thres_disc=varargin{1};
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training_features��ѵ������������  
%training_targets��ѵ�������������  
%test_features����������������   
%pruning����֦ϵ��  
%thres_disc:��ɢ������ֵ��>thres_disc�϶�Ϊ������ȡֵ��Χ���� 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [fea, num]     = size(train_features); %num��ѵ����������fea��������Ŀ  
    pruning    = pruning*num/100;  % ���ڼ�֦ 
        
    %�ж�ĳһά����������ɢȡֵ��������ȡֵ��0��������������
    discrete_dim =discreteOrContinue(train_features,thres_disc); 
        
    % �ݹ�ع�����  
    %disp('Building tree')    
    tree= build_tree(train_features, train_targets,discrete_dim,0,pruning);    
    save tree.mat tree;  
    %���뱯�ۼ�֦�Ĳ���  
    %����ȫ�����ľ������Ļ����ϣ������������Ч�����ѵ����������޼�����С�������ĸ��Ӷȣ����͹���ϵ�Ӱ��  
    %treeplot(tree);  
      
    %����Ԥ��    
    %disp('Classify test samples using the tree')    
    test_targets= predict(tree,test_features, 1:size(test_features,2), discrete_dim);    