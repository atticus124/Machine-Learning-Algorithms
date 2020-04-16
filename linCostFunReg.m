function [J,grad] = linCostFunReg(theta,X,y,lambda)
%Computes the cost function and gradient for logistic regression with regularization
    
%Variables:
%          theta: vector with paraemters for logistic regression
%          X:matrix with training examples; example i in row i
%          y: vector with output for training example i in row i
%          lambda: regularization parameter
    
%Output:
%          J= cost     
%          grad= vector of dJ/dtheta_j

    %if no regularization parameter passed, default to zero
    if nargin<4
       lambda=0;
    end
    
    
    m = length(y); % number of training example
    
    d=X*theta;
    modtheta=[0; theta(2:end)];
    J=(1/(2*m))*d*d'+(lambda/(2*m))*modtheta'*modtheta;
    grad = (1/m)*X'*(X*theta-y) +(lambda/m)*modtheta;
end
