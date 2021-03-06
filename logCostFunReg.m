function [J,grad] = logCostFunReg(X,y,theta,lambda)
%Computes the cost function and gradient for logistic regression with regularization
    
%Variables:
%          X:matrix with training examples; example i in row i
%                 NOTE: we assume that X has been prepended with a column of 1s
%          y: vector with output for training example i in row i
%          theta: vector with parameters for logistic regression
%          lambda: regularization parameter
    
    
    
%Output:
%          J= cost     
%          grad= vector of dJ/dtheta_j

    %if no regularization parameter passed, default to zero
    if nargin<4
       lambda=0;
    end
    
    
    m = length(y); % number of training example
    
   
    
    h=sigmoid(X*theta);
    modtheta=[0; theta(2:end)];
    J=(1/m)*(-y'*log(h)-(1-y')*log(1-h))+(lambda/(2*m))*modtheta'*modtheta;
    grad = (1/m)*X'*(h-y)+(lambda/m)*modtheta;
    
    
    
end
