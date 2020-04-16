function g = sigmoid(z)
% Computes the sigmoid function of z; If z is a matrix it computes the function 
%              componentwise
    
    g=  1./(1+exp(-z));

end