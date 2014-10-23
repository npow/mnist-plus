function w=logisticRegression(X,y,alpha,epsilon,numIter)

% solves for the weights using gradient descent for logistic regression by 
% taking the derivative and apply iteratively where,
% X is the n by m matrix of input data
% Y is the n by 1 vector of label
% alpha is the step size
% numIter is the number of iterations
% w is the m by 1 vector weights

[n,m]=size(X);
w=zeros(m,1); % initialize the weights
currentW=w; % initialize the weights

    for iIter=1:numIter % gradient descent       
        w=currentW-alpha.*X'*(sigmoid(X*currentW)-y)./n;
        if sum(abs(w-currentW)) < epsilon;
            break
        end
        currentW=w;
    end

end

function sig=sigmoid(x) % the logistic function (= sigmoid curve)
    sig=1./(1+exp(-x));
end