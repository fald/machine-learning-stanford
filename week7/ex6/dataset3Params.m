function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
range = [0.01 0.03 0.1 0.3 1 3 10 30];
results = ones(64, 3);
curr_row = 0;

for curr_C = range
  for curr_sigma = range
    curr_row += 1;
    disp(curr_row)
    %svmTrain(X, Y, C, kernelFunction, ...
     %                       tol, max_passes)
    %function sim = gaussianKernel(x1, x2, sigma)
    model = svmTrain(X, y, curr_C, @(x1, x2) gaussianKernel(x1, x2, curr_sigma));
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval));
    
    results(curr_row, :) = [curr_C curr_sigma prediction_error];
  end
end

final = sortrows(results, 3)(1, :); % For some reason using min fucked everything up, I am so mad
C = final(1)
sigma = final(2)
  



% =========================================================================

end
