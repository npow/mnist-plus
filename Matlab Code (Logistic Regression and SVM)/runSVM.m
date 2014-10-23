% run svm

pairwise                = nchoosek(0:9,2);            % 1-vs-1 pairwise models
svmModel                = cell(size(pairwise,1),1);            % Store binary-classifers
predTest                = zeros(size(testX,1),numel(svmModel)); % Store binary predictions

%# classify using one-against-one approach, linear SVM
for k=1:numel(svmModel)
    %# get only training instances belonging to this pair
    idx=any(bsxfun(@eq,trainy,pairwise(k,:)),2);

    %# train
    svmModel{k}=fitcsvm(trainX(idx,:),trainy(idx)); % linear SVM

    %# test
    predTest(:,k)=predict(svmModel{k}, testX);
    k
end
testy = mode(predTest,2);   % Voting: classify as the class receiving most votes