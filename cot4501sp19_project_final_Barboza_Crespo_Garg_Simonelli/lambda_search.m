function [ X ] = lambda_search( A, B, err_min, Train, Validation )

lambda=[A:(B-A)/1000:B];
    for j=1:1001
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
    left = left + x_i * x_i'+lambda(j);
    right = right + x_i * y_i';
end

W = inv(left) * right;

%% Validation and lambda search with minimal error
yRaw_Val = Validation(:,1);
xRaw_Val = Validation(:,2:end);
numInstances_Val=(size (Validation,1));

classes_Val = [];
for i = 1:numInstances_Val
    if ~ismember(yRaw_Val(i), classes_Val)
        classes_Val = [classes_Val; yRaw_Val(i)];
    end
end

numClasses_Val = length(classes_Val);

% create actual y matrix
y_Val = zeros(numInstances_Val, numClasses_Val);
for i = 1:numInstances_Val
    class_Val = yRaw_Val(i);
    y_Val(i, find(classes_Val==class_Val)) = 1;
end

[numInstances_Val, numFeatures_Val] = size(xRaw_Val);
[~, numClasses_Val] = size(y_Val);
numClassified_Val = numInstances_Val;

yPredicted = zeros(numClassified_Val, numClasses_Val);

numCorrect = 0;
err_valid=0;

% Check predicted values and search lambda with minimal error
for i =  1 : numInstances_Val
    predictionVector = (W' * xRaw_Val(i,:)')';
    [maxCol, colIndex] = max(predictionVector);
    yPredicted(i ,colIndex) = 1;
    
    [~, actualIndex] = max(y_Val(i,:));
    if (colIndex == actualIndex)
        numCorrect = numCorrect + 1;
    end
      err_valid=err_valid+(max(predictionVector)- max(y_Val(i,:))).^2;
end
corr(j)=numCorrect;
err(j)=err_valid;
 end
[a,b]=min(err);
d_err=err_min-a;
err_min=a;
    if b==1
    A=lambda(b);
    B=lambda(2);
    else
    if b==1001
    B=lambda(b);
    A=lambda(10);
    else
      A=lambda(b-1);
      B=lambda(b+1)*1.05;
    end
    end
    
    X.A=A;
    X.B=B;
    X.lambda=lambda;
    X.d_err=d_err;
    X.err=err;
    X.err_min=err_min;
    X.corr=corr;
    
end    