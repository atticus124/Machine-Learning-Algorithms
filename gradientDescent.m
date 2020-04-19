function [theta,J_history] = gradientDescent(X, y, theta, alpha, iterations, computeCost, lambda)
    
%Variables:
%          X:matrix with training examples; example i in row i
%                 NOTE: we assume that X has been prepended with a column of 1s
%          y: vector with output for training example i in row i
%          theta: vector with initial parameters for logistic regression
%          alpha: parameter for rate of descent
%          iterations: number of iterations to run descent
%          lambda: regularization parameter
    
    
    
    
    %if no regularization parameter passed, default to zero
    if nargin < 7
        lambda = 0;
    end
    
    
    for iter = 1:iterations
        [J,grad]=computeCost(X,y,theta,lambda);

        theta=theta-alpha*grad;
        J_history(iter) = J;
    end
    
end