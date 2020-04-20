function W = randInitialWeights(size_in, size_out)
%returns random matrix of size size_out x (size_in+1)
%to be used as inital Theta in running a neural net.
    epsilon= sqrt(6)/sqrt(size_out+size_in);    
    W = rand(size_out, 1 + size_in) *2*epsilon-epsilon;

end
