function discrete_dim=discreteOrContinue(train_features,thres_disc)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %�ж�ĳ��ά�ȵ������ǲ�������ȡֵ
    %train_features:ѵ����������
    %thres_disc:��ɢ������ֵ��>thres_disc�϶�Ϊ������ȡֵ��Χ����
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    fea=size(train_features,1);
    discrete_dim = zeros(1,fea);
    for i = 1:fea  %����ÿ������  
        Ub = unique(train_features(i,:)); 
        Nb = length(Ub);   
        if (Nb <= thres_disc)    
            discrete_dim(i) = Nb; %�õ�ѵ�������У�������������ظ�������ֵ����Ŀ �����discrete_dim(i)�У�i��ʾ��i������  
        end    
    end
end