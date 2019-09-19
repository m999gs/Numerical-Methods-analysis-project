 function [ X ] = classifier( filename, testpercent, lambda_scale )

%% Read data
Data = dlmread(filename);
trainPercent=100-testpercent;

%% Separation dataset to Training, Testing, Validation datasets

[m n] = size(Data);

%Make sure data is sorted by class
Data = sortrows(Data);

%% Generate test and training dataset
Test = [];
Train = [];
Validation=[];
i = 0;

while(i < m)
    first = i + 1;
    len = size(find(Data(:,1) == Data(first,1)),1);
    i = i + len;
    last = i;

% %% fixed (nonrandom) separation
%     trainInd=[1:1:round((trainPercent / 100) * len)];
%     testInd=[round((trainPercent / 100) * len)+1:1:len];
%     Train = [Train; removerows(Data(first:last,:), [testInd])];
%     Test = [Test; removerows(Data(first:last,:), trainInd )];
%     Validation = Test;

% random separation 
    if m>300
          [trainInd,valInd,testInd] = dividerand(len, 1 - testpercent/100, 0.1, testpercent/100-0.1); 
        Validation = [Validation; removerows(Data(first:last,:), [testInd trainInd])];
        Train = [Train; removerows(Data(first:last,:), [testInd valInd])];
        Test = [Test; removerows(Data(first:last,:), trainInd )];
    else
        [trainInd,testInd] = dividerand(len, 1 - testpercent/100, testpercent/100); 
        Train = [Train; removerows(Data(first:last,:), [testInd])];
        Test = [Test; removerows(Data(first:last,:), trainInd )];
        Validation = Test;
    end
end

%% Least square classification

err_min=1000;
d_err=1;
A0=-lambda_scale;
B0=lambda_scale;

%Right part of curve sarch
% Search for peak on error curve and take it coordinate as left border
A=A0;
B=B0;
Z=lambda_search( A, B, err_min, Train, Validation );
[c,d]=max(Z.err);
A=Z.lambda(d);

% Lambda search
while d_err>0.01
    Z=lambda_search( A, B, err_min, Train, Validation );
    A=Z.A;
    B=Z.B;
    d_err=Z.d_err;
    err=Z.err;
    err_min=Z.err_min;
end

% Final right lambda
A=A0;
B=B0;
[a,b]=min(err);
lambda_final_right=Z.lambda(b);
corr_final_right=Z.corr(b);

%Right part of curve sarch
% Search for peak on error curve and take it coordinate as right border
Z=lambda_search( A, B, err_min, Train, Validation );
[c,d]=max(Z.err);
B=Z.lambda(d);

% Lambda search
while d_err>0.01
    Z=lambda_search( A, B, err_min, Train, Validation );
    A=Z.A;
    B=Z.B;
    d_err=Z.d_err;
    err=Z.err;
    err_min=Z.err_min;
end

% Left Final lambda
[a,b]=min(err);
lambda_final_left=Z.lambda(b);
corr_final_left=Z.corr(b);

if corr_final_right>=corr_final_left
    lambda_final=lambda_final_right;
    corr_final=corr_final_right;
 else
    lambda_final=lambda_final_left;
    corr_final=corr_final_left;
end
% Mow we get optimal lambda and can estimate both methods

%% Training
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

% Compute W
% do each sum independently 
left = zeros(numFeatures_Train);
right = zeros(numFeatures_Train, numClasses_Train);
for i = 1:(size (Train,1))
    x_i = xRaw_Train(i,:)';
    y_i = y_Train(i,:)';
    left = left + x_i * x_i'+lambda_final;
    right = right + x_i * y_i';
end

W = inv(left) * right;

%% Testing 

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

% Check predicted values and search lambda wiyh minimal error
yPredicted = zeros(numClassified_Test, numClasses_Test);
numCorrect = 0;
for i =  1 : numInstances_Test
    predictionVector = (W' * xRaw_Test(i,:)')';
    [maxCol, colIndex] = max(predictionVector);
    yPredicted(i ,colIndex) = 1;
    
    [~, actualIndex] = max(y_Test(i,:));
    if (colIndex == actualIndex)
        numCorrect = numCorrect + 1;
    end
end
corr_test=numCorrect;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SVM classification

SVM_Mdl=fitcecoc(xRaw_Train, yRaw_Train);
yPredicted_SVM=predict (SVM_Mdl,xRaw_Test);
numCorrect_SVM = 0;
numClassified_Test=size(yRaw_Test,1);
for i=1:numClassified_Test
        if (yPredicted_SVM(i) == yRaw_Test(i))
        numCorrect_SVM = numCorrect_SVM + 1;
        end
end

%% Result
X=[trainPercent testpercent corr_test numClassified_Test corr_test/numClassified_Test*100 numCorrect_SVM numCorrect_SVM/numClassified_Test*100];
 
end




