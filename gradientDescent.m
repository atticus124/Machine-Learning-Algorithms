function [theta,J_history] = gradientDescent(X, y, theta, alpha, iterations, computeCost, lambda)
    if nargin < 7
        lambda = 0;
    end
    
    
    for iter = 1:iterations
        [J,grad]=computeCost(X,y,theta,lambda);

        theta=theta-alpha*grad;
        J_history(iter) = J;
    end
    
end