function feature_new = NormalizeFeature(feature_old)
%% Normaliziation to [-1, 1] NaN value will be set to 0
% features should be ranged by columns, which means each column for one feature
% HDU, Bocheng Wang 2018.10
%%
X = feature_old;
for column = 1:size(X, 2)
    singleColumn = X(:, column);
    yy = mapminmax(singleColumn')';
    X(:, column) = yy;
end
feature_new = X;