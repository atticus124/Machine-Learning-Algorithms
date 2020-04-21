function rand_elements = randElements(X,K)
% returns K randomly chosen rows of X in a matrix rand_elements
% K  must be less than or equal to the number of rows of X

    
    
    
    m = size(X,1);
    rand_elements = zeros(K,size(X,2));
    rand_indices = randperm(m,K)
    for i = 1:K
        rand_elements(i,:) = X(rand_indices(i),:);
    end
end

