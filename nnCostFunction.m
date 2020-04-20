function [J grad] = nnCostFunction(Theta_unrolled,...
                                   hidden_layer_sizes,...
                                   num_labels, ...
                                   X, y, lambda)

%Variables:

%          Theta_unrolled: vector with in the matrices of Theta parameters between layers unrolled;
%                 first layer to second layer first, then second to third, etc
%          hidden_layer_sizes: array of sizes of hidden layer; first hidden layer listed first
%          num_labels: number of labels in the data
%          X: data; example as rows
%          y: vector which has in pos i known label for row i of X.
%          lambda: regularization parameter

    
%Output:
%          J= cost     
%          grad= vector of dJ/dtheta_ij^k in same order as Theta_unrolled
    
    
    
    m = size(X, 1);
    num_features = size(X,2);
    J = 0;
    grad=[];
    
    Delta={};
    num_layers = length(hidden_layer_sizes)+1;
    
    
    layer_sizes = [num_features, hidden_layer_sizes,num_labels];
    
    A_cell={};
    Z_cell={};
    
    Theta_cell={};
    initial_index = 1;
    for i = 1:num_layers
        end_index = initial_index+(layer_sizes(i)+1)*layer_sizes(i+1)-1;
        Theta =  reshape(Theta_unrolled(initial_index:end_index),...
                         layer_sizes(i+1),...
                         layer_sizes(i)+1);
        Theta_cell{i}=Theta;
        initial_index = end_index+1;
    end
    
    
    y_expected = zeros(num_labels,m);
    for i = 1:m
        y_expected(y(i),i)=1;
    end
    
    
    
    %forward propogation to compute cost
    A=[ones(1,size(X,1)); X'];
    theta_begin_index=1;
    for i = 1:num_layers
        if i != 1
            A=[ones(1,size(Z,2)); sigmoid(Z)];
        end
        
        Theta= Theta_cell{i};
        
        
        Z= Theta*A;
        A_cell{i} = A;
        Z_cell{i+1} = Z;
        Theta_storage{i}=Theta;
    end
    y_predicted = sigmoid(Z);
    y_expected;
    
    J=sum(-dot(y_expected,log(y_predicted),1)-dot((1-y_expected),log(1-y_predicted),1))/m;
    
    
    
    %backward propogation
    delta = y_predicted-y_expected;
    for j = 1:num_layers
        i = num_layers +1-j;
        A = A_cell{i};
        Z = Z_cell{i};
        
        Theta= Theta_cell{i};
        modTheta=[zeros(size(Theta,1),1),Theta(:,2:end)];
        
        J+=(lambda/m/2)*sum(sum(modTheta.^2));
        
        grad = [((delta*A')/m + (lambda/m)*modTheta)(:);grad];

        if i != 1
            delta = (Theta'*delta)(2:end,:).*sigmoidGradient(Z);
        end    
    end
end

