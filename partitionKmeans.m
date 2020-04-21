function [centroids, closest_centroid] = partitionKmeans(X,K,max_iters)
%Variables:
%          X: matrix of data points; each row a data point
%          K: number of partions
    
%Output:
%          centroids: matrix of centroids; each row is a centroid
%          closest centroids: vector which in ith position has index
%                             corresponding to row of centroid closest to
%                             ith row of X

    if nargin<3
        max_iters=0;
    end
    
    
    centroids = randElements(X,K);
    iters = 1;
    

    
    a=true;
    while a
        closest_centroids = findClosestCentroids(X,centroids);
        partition_average = zeros(K,size(X,2));
        for i =1:K
            partition_average(i,:) = ...
                 mean(X((closest_centroids==i),:),1);
        end
        
        
        iters +=1;
        if (centroids == partition_average) || (iters == max_iters)
            a=false;
            

        end
        centroids = partition_average ;
    end
end