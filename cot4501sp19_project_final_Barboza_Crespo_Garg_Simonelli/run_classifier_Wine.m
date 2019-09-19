testpercents=[90 80 70 60 50];

cross_validation=10;   % iterations number of random subsampling

wine=classifier_table('wine.data', testpercents, cross_validation, 1)
 figure;
 plot(table2array(wine(:,1)),table2array(wine(:,5)));
 hold on
 plot(table2array(wine(:,1)),table2array(wine(:,7)));
title('wine');
xlabel('Training percent') 
ylabel('Correct prediction percent') 
legend ('Homegrown classifier','SVM classifier','Location','southeast')
grid on;