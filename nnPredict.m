function p = nnPredict(Theta, X)

%Variables:

%          Theta:ecell array of Theta values for different layers of neural net
%          X: input data. Individual entries as rows
    
%Output:
%          p= vector of predictions from running neural net with weights Theta
%                    on input X
    
    
% Useful values
m = size(X, 1);
num_layers= length(Theta);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h=X;
for i = 1:num_layers
    h = sigmoid([ones(m,1) h]*Theta{i}');
end
[dummy, p] = max(h, [], 2);

% =========================================================================


end
