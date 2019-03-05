% Data Analyze for HC, MCI and AD
% HDU, Bocheng Wang 2018.12

tic;
close all;
clear all;
clc;
addpath(genpath(pwd));


classes = {  '0.HC'; '1.EMCI'; '2.MCI'; '3.LMCI'; '4.AD';};
for index = 1:size(classes)
    path = './Data_back/BCTs/';
    files = dir(path);
    variblename = char(classes(index, 1));
    filename = [path variblename '.mat'];
    str = [variblename(3:size(variblename, 2)), '= load(filename)'];
    eval(str);    
end

subject_HC_AD = zeros(576, 360);
subject_HC_HC = zeros(576, 360);
index = 1;
for subject_HC_index = 1:size(HC.subjects)
    subject_HC = HC.subjects(subject_HC_index, 1082:1441);
    for subject_AD_index = 1:size(AD.subjects)        
        
        subject_AD = AD.subjects(subject_AD_index, 1082:1441);
        subject_HC_AD(index, :) = subject_HC - subject_AD;
        
        subject_HC_ = HC.subjects(subject_AD_index, 1082:1441); 
        subject_HC_HC(index, :) = subject_HC - subject_HC_;
        
        index = index + 1;
    end
end

%% Between Centrality
figure,  imagesc(abs(subject_HC_HC));title('HC-HC');
figure,  imagesc(abs(subject_HC_AD));title('HC-AD');

subject_HC_HC_Binarized = abs(subject_HC_HC) > 1;
subject_HC_AD_Binarized = abs(subject_HC_AD) > 1;

figure,  imagesc(subject_HC_HC_Binarized);title('HC-HC Thresholded (P = 0.5) and Binarized');
figure,  imagesc(subject_HC_AD_Binarized);title('HC-AD Thresholded (P = 0.5) and Binarized');

subject_HC_HC_sum = sum(subject_HC_HC_Binarized);
subject_HC_AD_sum = sum(subject_HC_AD_Binarized);

figure,  plot(subject_HC_HC_sum);title('HC-HC Binarize SUM for Each Area');
figure,  plot(subject_HC_AD_sum);title('HC-AD Binarize SUM for Each Area');

figure,  plot((subject_HC_AD_sum - subject_HC_HC_sum));title('AD-HC');


figure,  plot(abs(subject_HC_AD_sum - subject_HC_HC_sum)>40);title('AD vs HC significant difference in Betweenness Centrality');



toc;