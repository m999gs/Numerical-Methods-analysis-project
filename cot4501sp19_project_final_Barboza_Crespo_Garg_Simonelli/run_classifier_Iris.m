testpercents=[90 80 70 60 50];

cross_validation=10;   % iterations number of random subsampling

iris=classifier_table('iris_formatted.data', testpercents, cross_validation, 1)
figure;
plot(table2array( iris(:,1)),table2array( iris(:,5)));
hold on
plot(table2array( iris(:,1)),table2array( iris(:,7)),'g');
title(' iris');
xlabel('Training percent') 
ylabel('Correct prediction percent') 
legend ('Homegrown classifier','SVM classifier','Location','southeast')
grid on;