function accuracy=cal_accuracy(test_targets,test_targets_predict)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %��������������㷨��׼ȷ��
    %test_targets:�������ݼ���ʵ���������
    %test_targets_predict:�������㷨��Ԥ��Ĳ������ݼ����������
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    right=0;
    for j=1:size(test_targets_predict,2)  
        if test_targets(:,j)==test_targets_predict(:,j)  
            right=right+1;  
        end  
    end
    accuracy=right/size(test_targets,2);
end