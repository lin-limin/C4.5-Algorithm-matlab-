function targets = predict(tree,test_features, indices, discrete_dim)       
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����C4.5�������㷨�Բ�����������Ԥ��
%tree��C4.5�㷨�������ľ����� 
%test_features���������������� 
%indices������ 
%discrete:����ά�ȵ������Ƿ�������ȡֵ��0ָ��������ȡֵ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


targets = zeros(1, size(test_features,2)); 
        
if (tree.feature_tosplit == 0)  
    targets(indices) = tree.child;  %�õ�������Ӧ�ı�ǩ��tree.child  
    return    
end    
        
feature_tosplit = tree.feature_tosplit;  %�õ���������  
dims= 1:size(test_features,1);  %�õ���������  
        
% ���ݵõ��ľ������Բ����������з���  
if (discrete_dim(feature_tosplit) == 0) %�����ǰ���������Ǹ��������� 
    in= indices(find(test_features(feature_tosplit, indices)<= tree.location));  
    targets= targets + predict( tree.child(1),test_features(dims, :), in,discrete_dim(dims)); 
    in= indices(find(test_features(feature_tosplit, indices)>tree.location)); 
    targets= targets + predict(tree.child(2),test_features(dims, :),in,discrete_dim(dims));   
else  %�����ǰ���������Ǹ���ɢ����  
    Uf= unique(test_features(feature_tosplit,:)); %�õ������������������������ظ�����ֵ  
    for i = 1:length(Uf)  %����ÿ������ֵ    
        if any(Uf(i) == tree.value)  %tree.NfΪ���ķ����������� ��ǰ�����������������������ֵ  
            in= indices(find(test_features(feature_tosplit, indices) == Uf(i)));  %�ҵ���ǰ�����������������������ֵ==����ֵ����������  
            targets = targets + predict(tree.child(find(Uf(i)==tree.value)),test_features(dims, :),in,discrete_dim(dims));%���ⲿ�������ٷֲ�   
        end    
    end    
end 