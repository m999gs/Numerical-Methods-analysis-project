%% Read data

%Manual input
testpercent = 90;
filename='haberman.formatted.data';
Data = dlmread(filename);

%% Separation dataset to Train and Test

[m n] = size(Data);
trainPercent=100-testpercent;
%Make sure data is sorted by class
Data = sortrows(Data);

%Generate test and training dataset
Test = [];
Train = [];
Validation=[];
i = 0;

% while(i < m)
%     first = i + 1;
%     len = size(find(Data(:,1) == Data(first,1)),1);
%     i = i + len;
%     last = i;
% 
% %     %     fixed (nonrandom) separation
% %     trainInd=[1:1:round((trainPercent / 100) * len)];
% %     testInd=[round((trainPercent / 100) * len)+1:1:len];
% % random separation 
% %     if m>300
% %           [trainInd,valInd,testInd] = dividerand(len, 1 - testpercent/100, 0.1, testpercent/100-0.1); 
% %         Validation = [Validation; removerows(Data(first:last,:), [testInd trainInd])];
% %         Train = [Train; removerows(Data(first:last,:), [testInd valInd])];
% %         Test = [Test; removerows(Data(first:last,:), trainInd )];
% %     else
% %         [trainInd,testInd] = dividerand(len, 1 - testpercent/100, testpercent/100); 
% %         Train = [Train; removerows(Data(first:last,:), [testInd])];
% %         Test = [Test; removerows(Data(first:last,:), trainInd )];
% %         Validation = Test;
% %     end
% 
%     
%     Train = [Train; removerows(Data(first:last,:), testInd)];
%     Test = [Test; removerows(Data(first:last,:), trainInd)];
% 
% end

Train =[     1    33    58    10
     1    37    63     0
     1    39    58     0
     1    39    63     0
     1    39    63     4
     1    41    58     0
     1    49    63     3
     1    49    66     0
     1    50    61     6
     1    52    62     1
     1    53    60     1
     1    53    61     1
     1    54    58     1
     1    54    62     0
     1    57    62     0
     1    57    69     0
     1    59    60     0
     1    59    64     7
     1    60    67     2
     1    61    64     0
     1    61    65     8
     1    64    65    22
     1    30    65     0
     1    33    58    10
     1    34    61    10
     1    36    69     0
     1    37    63     0
     1    38    66    11
     1    39    58     0
     1    39    63     0
     1    39    63     4
     1    41    58     0
     1    43    60     0
     1    43    63     2
     1    45    67     0
     1    45    67     1
     1    49    63     3
     1    49    66     0
     1    50    59     2
     1    50    61     6
     1    50    64     0
     1    51    64     7
     1    52    62     1
     1    52    65     0
     1    53    60     1
     1    53    61     1
     1    54    58     1
     1    54    62     0
     1    55    58     1
     1    55    66     0
     1    55    66    18
     1    56    60     0
     1    57    62     0
     1    57    64     0
     1    57    64     0
     1    57    69     0
     1    59    60     0
     1    59    64     7
     1    59    67     3
     1    60    67     2
     1    61    64     0
     1    61    65     8
     1    61    68     0
     1    63    63     0
     1    64    65    22
     1    72    64     0
     1    76    67     0
     2    44    64     6
     2    45    67     1
     2    52    59     2
     2    57    64     1
     2    65    62    22
     2    70    58     0
     2    72    63     0
     2    74    65     3
     2    38    69    21
     2    43    59     2
     2    44    64     6
     2    45    67     1
     2    52    59     2
     2    53    58     4
     2    53    65     1
     2    54    60    11
     2    57    64     1
     2    63    60     1
     2    65    62    22
     2    66    58     0
     2    70    58     0
     2    72    63     0
     2    74    65     3
     2    83    58     2];
 
 Test =[1    30    62     3
     1    30    64     1
     1    30    65     0
     1    31    59     2
     1    31    65     4
     1    33    60     0
     1    34    58    30
     1    34    60     0
     1    34    60     1
     1    34    61    10
     1    34    67     7
     1    35    63     0
     1    35    64    13
     1    36    60     1
     1    36    69     0
     1    37    58     0
     1    37    59     6
     1    37    60     0
     1    37    60    15
     1    37    63     0
     1    38    59     2
     1    38    60     0
     1    38    60     0
     1    38    60     1
     1    38    62     3
     1    38    64     1
     1    38    66     0
     1    38    66    11
     1    38    67     5
     1    39    59     2
     1    39    67     0
     1    40    58     0
     1    40    58     2
     1    40    65     0
     1    41    59     0
     1    41    59     8
     1    41    64     0
     1    41    65     0
     1    41    65     0
     1    41    69     8
     1    42    58     0
     1    42    59     2
     1    42    60     1
     1    42    61     4
     1    42    62    20
     1    42    63     1
     1    42    65     0
     1    43    60     0
     1    43    63     2
     1    43    63    14
     1    43    64     2
     1    43    64     3
     1    43    65     0
     1    43    66     4
     1    44    61     0
     1    44    61     0
     1    44    63     1
     1    44    67    16
     1    45    59    14
     1    45    60     0
     1    45    64     0
     1    45    67     0
     1    45    67     1
     1    45    68     0
     1    46    58     3
     1    46    62     0
     1    46    63     0
     1    47    58     3
     1    47    60     4
     1    47    61     0
     1    47    63     6
     1    47    66     0
     1    47    66    12
     1    47    67     0
     1    47    68     4
     1    48    61     8
     1    48    62     2
     1    48    64     0
     1    48    66     0
     1    49    60     1
     1    49    61     0
     1    49    61     1
     1    49    62     0
     1    49    62     1
     1    49    67     1
     1    50    58     1
     1    50    59     0
     1    50    59     2
     1    50    61     0
     1    50    61     0
     1    50    63     1
     1    50    64     0
     1    50    65     4
     1    50    66     1
     1    51    59     1
     1    51    64     7
     1    51    65     0
     1    51    66     1
     1    52    60     4
     1    52    60     5
     1    52    61     0
     1    52    62     0
     1    52    63     4
     1    52    64     0
     1    52    65     0
     1    52    68     0
     1    52    69     0
     1    53    58     1
     1    53    60     2
     1    53    63     0
     1    54    59     7
     1    54    60     3
     1    54    62     0
     1    54    63    19
     1    54    66     0
     1    54    67    46
     1    54    69     7
     1    55    58     0
     1    55    58     1
     1    55    58     1
     1    55    66     0
     1    55    66    18
     1    55    67     1
     1    55    69     3
     1    55    69    22
     1    56    60     0
     1    56    60     0
     1    56    66     1
     1    56    66     2
     1    56    67     0
     1    57    61     0
     1    57    63     0
     1    57    64     0
     1    57    64     0
     1    57    64     9
     1    57    67     0
     1    58    58     0
     1    58    58     3
     1    58    59     0
     1    58    60     3
     1    58    61     1
     1    58    61     2
     1    58    67     0
     1    59    63     0
     1    59    64     0
     1    59    64     1
     1    59    64     4
     1    59    67     3
     1    60    61     1
     1    60    61    25
     1    60    64     0
     1    61    59     0
     1    61    59     0
     1    61    59     0
     1    61    68     0
     1    62    58     0
     1    62    62     6
     1    62    66     0
     1    62    66     0
     1    63    61     0
     1    63    61     9
     1    63    61    28
     1    63    62     0
     1    63    63     0
     1    63    63     0
     1    63    66     0
     1    64    58     0
     1    64    61     0
     1    64    66     0
     1    64    68     0
     1    65    58     0
     1    65    59     2
     1    65    64     0
     1    65    64     0
     1    65    67     0
     1    65    67     1
     1    66    58     0
     1    66    58     1
     1    66    68     0
     1    67    61     0
     1    67    65     0
     1    67    66     0
     1    67    66     0
     1    68    67     0
     1    68    68     0
     1    69    60     0
     1    69    65     0
     1    69    66     0
     1    70    59     8
     1    70    63     0
     1    70    66    14
     1    70    67     0
     1    70    68     0
     1    71    68     2
     1    72    58     0
     1    72    64     0
     1    72    67     3
     1    73    62     0
     1    73    68     0
     1    74    63     0
     1    75    62     1
     1    76    67     0
     1    77    65     3
     1    30    62     3
     1    30    64     1
     1    30    65     0
     1    31    59     2
     1    31    65     4
     1    33    60     0
     1    34    58    30
     1    34    60     0
     1    34    60     1
     1    34    61    10
     1    34    67     7
     1    35    63     0
     1    35    64    13
     1    36    60     1
     1    36    69     0
     1    37    58     0
     1    37    59     6
     1    37    60     0
     1    37    60    15
     1    37    63     0
     1    38    59     2
     1    38    60     0
     1    38    60     0
     1    38    60     1
     1    38    62     3
     1    38    64     1
     1    38    66     0
     1    38    66    11
     1    38    67     5
     1    39    59     2
     1    39    67     0
     1    40    58     0
     1    40    58     2
     1    40    65     0
     1    41    59     0
     1    41    59     8
     1    41    64     0
     1    41    65     0
     1    41    65     0
     1    41    69     8
     1    42    58     0
     1    42    59     2
     1    42    60     1
     1    42    61     4
     1    42    62    20
     1    42    63     1
     1    42    65     0
     1    43    60     0
     1    43    63     2
     1    43    63    14
     1    43    64     2
     1    43    64     3
     1    43    65     0
     1    43    66     4
     1    44    61     0
     1    44    61     0
     1    44    63     1
     1    44    67    16
     1    45    59    14
     1    45    60     0
     1    45    64     0
     1    45    67     0
     1    45    67     1
     1    45    68     0
     1    46    58     3
     1    46    62     0
     1    46    63     0
     1    47    58     3
     1    47    60     4
     1    47    61     0
     1    47    63     6
     1    47    66     0
     1    47    66    12
     1    47    67     0
     1    47    68     4
     1    48    61     8
     1    48    62     2
     1    48    64     0
     1    48    66     0
     1    49    60     1
     1    49    61     0
     1    49    61     1
     1    49    62     0
     1    49    62     1
     1    49    67     1
     1    50    58     1
     1    50    59     0
     1    50    59     2
     1    50    61     0
     1    50    61     0
     1    50    63     1
     1    50    64     0
     1    50    65     4
     1    50    66     1
     1    51    59     1
     1    51    64     7
     1    51    65     0
     1    51    66     1
     1    52    60     4
     1    52    60     5
     1    52    61     0
     1    52    62     0
     1    52    63     4
     1    52    64     0
     1    52    65     0
     1    52    68     0
     1    52    69     0
     1    53    58     1
     1    53    60     2
     1    53    63     0
     1    54    59     7
     1    54    60     3
     1    54    62     0
     1    54    63    19
     1    54    66     0
     1    54    67    46
     1    54    69     7
     1    55    58     0
     1    55    58     1
     1    55    58     1
     1    55    66     0
     1    55    66    18
     1    55    67     1
     1    55    69     3
     1    55    69    22
     1    56    60     0
     1    56    60     0
     1    56    66     1
     1    56    66     2
     1    56    67     0
     1    57    61     0
     1    57    63     0
     1    57    64     0
     1    57    64     0
     1    57    64     9
     1    57    67     0
     1    58    58     0
     1    58    58     3
     1    58    59     0
     1    58    60     3
     1    58    61     1
     1    58    61     2
     1    58    67     0
     1    59    63     0
     1    59    64     0
     1    59    64     1
     1    59    64     4
     1    59    67     3
     1    60    61     1
     1    60    61    25
     1    60    64     0
     1    61    59     0
     1    61    59     0
     1    61    59     0
     1    61    68     0
     1    62    58     0
     1    62    62     6
     1    62    66     0
     1    62    66     0
     1    63    61     0
     1    63    61     9
     1    63    61    28
     1    63    62     0
     1    63    63     0
     1    63    63     0
     1    63    66     0
     1    64    58     0
     1    64    61     0
     1    64    66     0
     1    64    68     0
     1    65    58     0
     1    65    59     2
     1    65    64     0
     1    65    64     0
     1    65    67     0
     1    65    67     1
     1    66    58     0
     1    66    58     1
     1    66    68     0
     1    67    61     0
     1    67    65     0
     1    67    66     0
     1    67    66     0
     1    68    67     0
     1    68    68     0
     1    69    60     0
     1    69    65     0
     1    69    66     0
     1    70    59     8
     1    70    63     0
     1    70    66    14
     1    70    67     0
     1    70    68     0
     1    71    68     2
     1    72    58     0
     1    72    64     0
     1    72    67     3
     1    73    62     0
     1    73    68     0
     1    74    63     0
     1    75    62     1
     1    76    67     0
     1    77    65     3
     2    34    59     0
     2    34    66     9
     2    38    69    21
     2    39    66     0
     2    41    60    23
     2    41    64     0
     2    41    67     0
     2    42    59     0
     2    42    69     1
     2    43    58    52
     2    43    59     2
     2    43    64     0
     2    43    64     0
     2    44    58     9
     2    44    63    19
     2    45    65     6
     2    45    66     0
     2    46    58     2
     2    46    62     5
     2    46    65    20
     2    46    69     3
     2    47    62     0
     2    47    63    23
     2    47    65     0
     2    48    58    11
     2    48    58    11
     2    48    67     7
     2    49    63     0
     2    49    64    10
     2    50    63    13
     2    50    64     0
     2    51    59     3
     2    51    59    13
     2    52    62     3
     2    52    66     4
     2    52    69     3
     2    53    58     4
     2    53    59     3
     2    53    60     9
     2    53    63    24
     2    53    65     1
     2    53    65    12
     2    54    60    11
     2    54    65     5
     2    54    65    23
     2    54    68     7
     2    55    63     6
     2    55    68    15
     2    56    65     9
     2    56    66     3
     2    57    61     5
     2    57    62    14
     2    59    62    35
     2    60    59    17
     2    60    65     0
     2    61    62     5
     2    61    65     0
     2    61    68     1
     2    62    58     0
     2    62    59    13
     2    62    65    19
     2    63    60     1
     2    65    58     0
     2    65    61     2
     2    65    66    15
     2    66    58     0
     2    66    61    13
     2    67    63     1
     2    67    64     8
     2    69    67     8
     2    70    58     4
     2    78    65     1
     2    83    58     2
     2    34    59     0
     2    34    66     9
     2    38    69    21
     2    39    66     0
     2    41    60    23
     2    41    64     0
     2    41    67     0
     2    42    59     0
     2    42    69     1
     2    43    58    52
     2    43    59     2
     2    43    64     0
     2    43    64     0
     2    44    58     9
     2    44    63    19
     2    45    65     6
     2    45    66     0
     2    46    58     2
     2    46    62     5
     2    46    65    20
     2    46    69     3
     2    47    62     0
     2    47    63    23
     2    47    65     0
     2    48    58    11
     2    48    58    11
     2    48    67     7
     2    49    63     0
     2    49    64    10
     2    50    63    13
     2    50    64     0
     2    51    59     3
     2    51    59    13
     2    52    62     3
     2    52    66     4
     2    52    69     3
     2    53    58     4
     2    53    59     3
     2    53    60     9
     2    53    63    24
     2    53    65     1
     2    53    65    12
     2    54    60    11
     2    54    65     5
     2    54    65    23
     2    54    68     7
     2    55    63     6
     2    55    68    15
     2    56    65     9
     2    56    66     3
     2    57    61     5
     2    57    62    14
     2    59    62    35
     2    60    59    17
     2    60    65     0
     2    61    62     5
     2    61    65     0
     2    61    68     1
     2    62    58     0
     2    62    59    13
     2    62    65    19
     2    63    60     1
     2    65    58     0
     2    65    61     2
     2    65    66    15
     2    66    58     0
     2    66    61    13
     2    67    63     1
     2    67    64     8
     2    69    67     8
     2    70    58     4
     2    78    65     1
     2    83    58     2];

%% Least square classification
lambda=[-1:0.01:1]*500;
corr=[];
err=[];
err=[];

for j=1:size(lambda,2)

%%% Training

yRaw_Train = Train(:,1);
xRaw_Train = Train(:,2:end);

[numInstances_Train, numFeatures_Train] = size(xRaw_Train);

classes_Train = [];
for i = 1:numInstances_Train
    if ~ismember(yRaw_Train(i), classes_Train)
        classes_Train = [classes_Train; yRaw_Train(i)];
    end
end

numClasses_Train = length(classes_Train);

% create actual y matrix
y_Train = zeros(numInstances_Train, numClasses_Train);
for i = 1:numInstances_Train
    class_Train = yRaw_Train(i);
    y_Train(i, find(classes_Train==class_Train)) = 1;
end

%Compute W
% do each sum independently 
left = zeros(numFeatures_Train);
right = zeros(numFeatures_Train, numClasses_Train);
for i = 1:(size (Train,1))
    x_i = xRaw_Train(i,:)';
    y_i = y_Train(i,:)';
    left = left + x_i * x_i'+lambda(j);
    right = right + x_i * y_i';
end

W = inv(left) * right;



%%% Testing 

yRaw_Test = Test(:,1);
xRaw_Test = Test(:,2:end);
numInstances_Test=(size (Test,1));

classes_Test = [];
for i = 1:numInstances_Test
    if ~ismember(yRaw_Test(i), classes_Test)
        classes_Test = [classes_Test; yRaw_Test(i)];
    end
end

numClasses_Test = length(classes_Test);

% create actual y matrix
y_Test = zeros(numInstances_Test, numClasses_Test);
for i = 1:numInstances_Test
    class_Test = yRaw_Test(i);
    y_Test(i, find(classes_Test==class_Test)) = 1;
end

[numInstances_Test, numFeatures_Test] = size(xRaw_Test);
[~, numClasses_Test] = size(y_Test);
numClassified_Test = numInstances_Test;

yPredicted = zeros(numClassified_Test, numClasses_Test);

numCorrect = 0;
err_valid=0;
aaa=[];

for i =  1 : numInstances_Test
    predictionVector = (W' * xRaw_Test(i,:)')';
    
    [maxCol, colIndex] = max(predictionVector);
    yPredicted(i ,colIndex) = 1;
    
    [~, actualIndex] = max(y_Test(i,:));
    if (colIndex == actualIndex)
        numCorrect = numCorrect + 1;
    end
     err_valid=err_valid+(max(predictionVector)- max(y_Test(i,:))).^2;
end
corr(j)=numCorrect;
err(j)=err_valid;
end


plot(lambda,corr/numClassified_Test*100);
xlabel('lambda');
ylabel('correct predictions,%');
grid on
figure

plot(lambda,err);
xlabel('lambda');
ylabel('error');
grid on


