% Classification for HC, MCI and AD
% HDU, Bocheng Wang 2018.10

tic;
clear all;
clc;
addpath(genpath(pwd));

%% Parameters 
% filterFS: { 'Rank',  'Predefined', 'Predefined_and_Rank'}

% Rank for features selection without any predefined set. Filter and
%                      wrapper selection were both carried out.

% Predefined for features parse. Features were selected previously. In this
%                      way, filter and wrapper selection were both omitted.

% Predefined_and_Rank for preset some core features and keep them as
%                      necessary features in wrapper procedure. In this
%                      way, filter selection, rank importance, was omitted.

filterFS = 'Predefined_and_Rank';

% Core features set with P < 0.05
% FilterdIndex = [486 88 808 1528 2248 746 1653 1330 2608 4048 2733 3963];

% Core features set with P < 0.1
FilterdIndex = [88 808 1528 2248 1780 746 1623 1653 1330 699 1445 2608 4048 2733 4173 3903 3219 4019 3668 4143 3823 3874 3884 3963 1025 1040 486 1437 4458 3492 5379];



%% load data
load('./Data_with_HC=24/BCTs/0.HC.mat');
Subject_HC = subjects;

load('./Data_with_HC=24/BCTs/1.EMCI.mat');
Subject_EMCI = subjects;

load('./Data_with_HC=24/BCTs/2.MCI.mat');
Subject_MCI = subjects;

load('./Data_with_HC=24/BCTs/3.LMCI.mat');
Subject_LMCI = subjects;

load('./Data_with_HC=24/BCTs/4.AD.mat');
Subject_AD = subjects;

HC_vs_EMCI   = cat(1, Subject_HC, Subject_EMCI);

% for index = 1:size(HC_vs_EMCI_vs_LMCI_vs_AD, 1)
%     if HC_vs_EMCI_vs_LMCI_vs_AD(index, 1) == 1000
%         HC_vs_EMCI_vs_LMCI_vs_AD(index, 1) = 10;
%     end
% end

HC_vs_EMCI_vs_LMCI_AD = cat(1, Subject_AD, Subject_EMCI);
HC_vs_EMCI_vs_LMCI_AD = cat(1, HC_vs_EMCI_vs_LMCI_AD, Subject_LMCI);
HC_vs_EMCI_vs_LMCI_AD = cat(1, HC_vs_EMCI_vs_LMCI_AD, Subject_HC);
trainingSet = HC_vs_EMCI_vs_LMCI_AD;


% delete subgraph centrality
trainingSet(:, 4336:4695) = [];
for index = 2:size(trainingSet, 2)
    trainingSet(:, index) = mapminmax(trainingSet(:, index)')';
end




%% Filter Feature selection
if strcmp(filterFS, 'Rank')
    [FilteredMatrix, FilterdIndex] = Filter_Feature_Rank_importance(trainingSet, 1/5);
elseif strcmp(filterFS, 'Predefined')    
    X = trainingSet(:, 2:size(trainingSet, 2));
    y = trainingSet(:, 1);    
    FilteredMatrix = [y X(:, FilterdIndex)];
elseif strcmp(filterFS, 'Predefined_and_Rank')
    X = trainingSet(:, 2:size(trainingSet, 2));
    % for predefined feature set
    X_predefined = trainingSet(:, 2:size(trainingSet, 2));
    y = trainingSet(:, 1);    
    FilteredMatrix_predefined =  X(:, FilterdIndex);
    trainSet_for_coreFeatures = [FilteredMatrix_predefined y];
    % for filter feature selection
    [FilteredMatrix_all, FilterdIndex_all] = Filter_Feature_Rank_importance(trainingSet, 1/5);
    
    % combine (predefined set) with (filter feature selection set)
    FilteredMatrix = [FilteredMatrix_all FilteredMatrix_predefined ];
    
end
%% Wrapper Feature selection
if strcmp(filterFS, 'Rank')
    [Selected_train_data, SelectedFeatures_in_RankImportanceOrder] ...
    = WrapperFeatureSelection(FilteredMatrix, false, 0);
    RankImportanceOrder_2_FeatureName(FilterdIndex, SelectedFeatures_in_RankImportanceOrder);
elseif strcmp(filterFS, 'Predefined_and_Rank')
    [Selected_train_data, SelectedFeatures_in_RankImportanceOrder] ...
    = WrapperFeatureSelection(FilteredMatrix, false, 1:size(FilteredMatrix_predefined, 2));
    RankImportanceOrder_2_FeatureName(FilterdIndex_all, SelectedFeatures_in_RankImportanceOrder(SelectedFeatures_in_RankImportanceOrder <= round((size(trainingSet, 2) -1) * 0.2) ));
    coreFeaturesSelected = SelectedFeatures_in_RankImportanceOrder(SelectedFeatures_in_RankImportanceOrder > round((size(trainingSet, 2) -1) * 0.2));
    if size(coreFeaturesSelected, 2)
        RankImportanceOrder_2_FeatureName(FilterdIndex, coreFeaturesSelected - 1083);
    end
elseif strcmp(filterFS, 'Predefined')
    RankImportanceOrder_2_FeatureName(FilterdIndex, 1:size(FilterdIndex, 2));
    Selected_train_data = FilteredMatrix;
end

%% Matlab Machine learning Toolbox ...
%% libSVM tools
libSVM_result_filename = 'tempfiles\libSVM_result.txt';
matrix2libsvmformat(Selected_train_data, libSVM_result_filename);
libSVM_Accuracy_Output = evaluateSVM(libSVM_result_filename);
toc;
