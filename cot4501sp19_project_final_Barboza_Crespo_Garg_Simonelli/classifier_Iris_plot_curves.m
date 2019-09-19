%% Read data

%Manual input
testpercent = 90;
filename='iris-number-classes.data';
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

while(i < m)
    first = i + 1;
    len = size(find(Data(:,1) == Data(first,1)),1);
    i = i + len;
    last = i;

    %     fixed (nonrandom) separation
    trainInd=[1:1:round((trainPercent / 100) * len)];
    testInd=[round((trainPercent / 100) * len)+1:1:len];
    
    Train = [Train; removerows(Data(first:last,:), testInd)];
    Test = [Test; removerows(Data(first:last,:), trainInd)];

end

%% Least square classification
lambda=[-1:0.01:1]/10;
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


