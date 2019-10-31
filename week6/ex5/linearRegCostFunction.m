function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%X1 = [ones(m, 1) X];
theta1 = theta(2:end, :);
h = X * theta;       
error = (h - y)' * (h - y);
reg = (lambda / (2 * m)) * (theta1' * theta1);
J = reg + error / (2 * m);

base_grad = (h - y)' * X;           % (m x 1)' * m x n -> 1 x n
reg_grad = [0; theta1];             % m x n;
grad = (base_grad' + lambda * reg_grad) / m;






% =========================================================================

grad = grad(:);

end
