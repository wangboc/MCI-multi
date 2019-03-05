function [FilteredMatrix, FilterdIndex] =  Feature_Rank_importance(Matrix, p)
%% Implement Filter Feature Selection algorithm
%  HDU, Bocheng Wang 2018.10
%%
X = Matrix(:, 2:size(Matrix, 2));
Y = Matrix(:, 1);

[ranks,weights] = relieff(X,Y,10, 'method','classification');
% [ranks,weights] = relieff(X,Y,10);
selected_columns = ranks(1:round(size(ranks, 2) * p));
FilteredMatrix = [Y X(:, selected_columns)];
FilterdIndex = selected_columns;
end