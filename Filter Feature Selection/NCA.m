function [FilteredMatrix, FilterdIndex] = NCA(Matrix,inputArg2)
%NCA Feature selection using neighborhood component analysis for classification
%  
X = Matrix(:, 2:size(Matrix, 2));
y = Matrix(:, 1);
mdl = fscnca(X,y);
figure()
plot(mdl.FeatureWeights,'ro')
grid on
xlabel('Feature index')
ylabel('Feature weight')
selected_columns = 1:size(X,2);
selected_columns = selected_columns(mdl.FeatureWeights>0.001);
FilteredMatrix = [y X(:, selected_columns)];
FilterdIndex = selected_columns;
end

