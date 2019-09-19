function [ T ] = classifier_table( filename, testpercents, cross_validation, lambda_scale )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

test_result_mean=[];
for i=1:size(testpercents,2)
    test_result=[];
    for j=1:cross_validation
    test_result=[test_result;classifier(filename, testpercents(i), lambda_scale)];
    end
    
    test_result_mean=[test_result_mean;mean(test_result,1)];
end

T = array2table(test_result_mean,'VariableNames',{'Percent_train','Percent_test','Correct_test','Total_test','Correct_percent','Correct_test_SVM','Correct_percent_SVM'});

end
