function  RankImportanceOrder_2_FeatureName(FilterdIndex, SelectedFeatures_in_RankImportanceOrder)

% for test   SelectedFeatures_in_RankImportanceOrder = [10 234 396 546 692 ];
%% Bocheng Wang 2019.03.03 update for multi-recognition for MCIs

features_index = FilterdIndex(SelectedFeatures_in_RankImportanceOrder);
for index = 1:size(features_index, 2)
    feature = features_index(index);
    [name, area] = ParseFeature(feature);
    disp(['����ID��' num2str(feature) ' �������ͣ�' name '  �������ڷ�����' char(area)]);
end

