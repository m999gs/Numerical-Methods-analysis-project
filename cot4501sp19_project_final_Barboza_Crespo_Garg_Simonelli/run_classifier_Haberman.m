testpercents=[90 80 70 60 50];

cross_validation=10;   % iterations number of random subsampling

haberman=classifier_table('haberman_formatted.data', testpercents, cross_validation, 500)
 figure;
 plot(table2array(haberman(:,1)),table2array(haberman(:,5)));
 hold on
 plot(table2array(haberman(:,1)),table2array(haberman(:,7)),'g');
title('haberman');
xlabel('Training percent') 
ylabel('Correct prediction percent') 
legend ('Homegrown classifier','SVM classifier','Location','southeast')
grid on;


