testpercents=[90 80 70 60 50];

cross_validation=10;   % iterations number of random subsampling

glass=classifier_table('glass_formatted.data', testpercents, cross_validation, 1)
 figure;
 plot(table2array(glass(:,1)),table2array(glass(:,5)));
 hold on
 plot(table2array(glass(:,1)),table2array(glass(:,7)),'g');
title('glass');
xlabel('Training percent') 
ylabel('Correct prediction percent') 
legend ('Homegrown classifier','SVM classifier','Location','southeast')
grid on;
