function g = sigmoidDeriv(z)
%component wise computes the derivative of the sigmoid function at z
    
g = zeros(size(z));

g=g+exp(z)./(1+exp(z)).^2;








% =============================================================




end
