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
HC_vs_MCI    = cat(1, Subject_HC, Subject_MCI);
HC_vs_LMCI   = cat(1, Subject_HC, Subject_LMCI);
HC_vs_AD     = cat(1, Subject_HC, Subject_AD);

EMCI_vs_MCI  = cat(1, Subject_EMCI, Subject_MCI);
EMCI_vs_LMCI = cat(1, Subject_EMCI, Subject_LMCI);
EMCI_vs_AD   = cat(1, Subject_EMCI, Subject_AD);

MCI_vs_LMCI  = cat(1, Subject_MCI, Subject_LMCI);
MCI_vs_AD    = cat(1, Subject_MCI, Subject_AD);

LMCI_vs_AD   = cat(1, Subject_LMCI, Subject_AD);

HC_vs_MCI_AD = cat(1, HC_vs_MCI, Subject_AD);

% HC_vs_EMCI_vs_LMCI_vs_AD = cat(1, HC_vs_EMCI, LMCI_vs_AD);




% for index = 1:size(HC_vs_EMCI_vs_LMCI_vs_AD, 1)
%     if HC_vs_EMCI_vs_LMCI_vs_AD(index, 1) == 1000
%         HC_vs_EMCI_vs_LMCI_vs_AD(index, 1) = 10;
%     end
% end

HC_vs_EMCI = cat(1, Subject_HC, Subject_EMCI);

% delete subgraph centrality
HC_vs_EMCI(:, 4336:4695) = [];
for index = 2:size(HC_vs_EMCI, 2)
    HC_vs_EMCI(:, index) = mapminmax(HC_vs_EMCI(:, index)')';
end



%% Filter Feature selection
if strcmp(filterFS, 'Rank')
    [FilteredMatrix, FilterdIndex] = Filter_Feature_Rank_importance(HC_vs_EMCI, 1/5);
elseif strcmp(filterFS, 'Predefined')    
    X = HC_vs_EMCI(:, 2:size(HC_vs_EMCI, 2));
    y = HC_vs_EMCI(:, 1);    
    FilteredMatrix = [y X(:, FilterdIndex)];
elseif strcmp(filterFS, 'Predefined_and_Rank')
    X = HC_vs_EMCI(:, 2:size(HC_vs_EMCI, 2));
    y = HC_vs_EMCI(:, 1);    
    FilteredMatrix = [y X];    
end
%% Wrapper Feature selection
if strcmp(filterFS, 'Rank')
    [Selected_train_data, SelectedFeatures_in_RankImportanceOrder] ...
    = WrapperFeatureSelection(FilteredMatrix);
    RankImportanceOrder_2_FeatureName(FilterdIndex, SelectedFeatures_in_RankImportanceOrder);
elseif strcmp(filterFS, 'Predefined_and_Rank')
    [Selected_train_data, SelectedFeatures_in_RankImportanceOrder] ...
    = WrapperFeatureSelection(FilteredMatrix);
    RankImportanceOrder_2_FeatureName(SelectedFeatures_in_RankImportanceOrder, 1:size(SelectedFeatures_in_RankImportanceOrder, 2));
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
