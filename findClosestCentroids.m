function closest_centroids = findClosestCentroids(X, centroids)
%Variables:
    
%          X: matrix of data points; each row a data point
%          centroids: matrix storing centroids; each row a centroid
    
%Output:
%          closestCentroids: vector where element i the index of closest 
%                           centroid to data point in row i of X



% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
closest_centroid = zeros(size(X,1), 1);

d=zeros(size(X,1),K); 

for i = 1:K
    d(:,i) = sum((X-centroids(i,:)).^2,2);
end

[min_dist, closest_centroids]=min(d,[],2);

end

