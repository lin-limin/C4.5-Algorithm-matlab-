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
%training_features：训练样本的特征  
%training_targets：训练样本所属类别  
%test_features：测试样本的特征   
%pruning：剪枝系数  
%thres_disc:离散特征阈值，>thres_disc认定为该特征取值范围连续 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [fea, num]     = size(train_features); %num是训练样本数，fea是特征数目  
    pruning    = pruning*num/100;  % 用于剪枝 
        
    %判断某一维的特征是离散取值还是连续取值，0代表是连续特征
    discrete_dim =discreteOrContinue(train_features,thres_disc); 
        
    % 递归地构造树  
    %disp('Building tree')    
    tree= build_tree(train_features, train_targets,discrete_dim,0,pruning);    
    save tree.mat tree;  
    %加入悲观剪枝的操作  
    %在完全生长的决策树的基础上，对生长后分类效果不佳的子树进行修剪，减小决策树的复杂度，降低过拟合的影响  
    %treeplot(tree);  
      
    %样本预测    
    %disp('Classify test samples using the tree')    
    test_targets= predict(tree,test_features, 1:size(test_features,2), discrete_dim);    