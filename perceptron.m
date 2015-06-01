function [w, iter] = perceptron(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE
[P N] = size(X);
X1 = [ones(1,N); X];
w = zeros(P+1,1);
iter = 0;
while (1)
    iter = iter + 1;
    for i = 1:N
        f = w'*X1(:,i);
        if (f >= 0)
            f = 1;
        else
            f = -1;
        end
        w = w  + 0.05*(y(i) - f)*X1(:,i);
    end
    h = y .* (w'*X1);
    if (sum(h<=0) == 0)
        break;
    end
end

