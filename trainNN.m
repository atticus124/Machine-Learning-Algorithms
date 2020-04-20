function [Theta, cost] = trainNN(minFunction,hidden_layer_sizes, num_labels, X,y, lambda, options)
%Variables:

%          minFunction: which finds values of weights to minimize cost function
%          hidden_layer_sizes: array of sizes of hidden layer; first hidden layer listed first
%          num_labels: number of labels in the data
%          X: data; example as rows
%          y: vector which has in pos i known label for row i of X.
%          lambda: regularization parameter
%          options: options for the minFunction

    
%Output:
%          Theta= cell array of matrix of weights; in position i matrix
%                   which transitions from ith layer to (i+1)st
%          cost = final cost on the training set
    

    num_features = size(X,2);
    num_layers = length(hidden_layer_sizes)+1;
    layer_sizes = [num_features hidden_layer_sizes num_labels];
    Theta = {};
    initial_params=[];
    
    for i = 1:num_layers
       rand_Theta=randInitialWeights(layer_sizes(i),layer_sizes(i+1)); 
       initial_params = [initial_params ; rand_Theta(:)];
    end
    
    
    length(initial_params)
    costFunction = @(p) nnCostFunction(p,hidden_layer_sizes,num_labels,X,y,lambda);
    [params,cost] = minFunction(costFunction,initial_params,options);
    
    start_index = 1;
    for i = 1:num_layers
        end_index = start_index + (layer_sizes(i+1))*(layer_sizes(i)+1)-1;
        out_Theta=reshape(params(start_index:end_index),...
                         layer_sizes(i+1), layer_sizes(i)+1);
        Theta{i}=out_Theta;
        start_index = end_index+1;        
    end
end