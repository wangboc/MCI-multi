function [SelectedTrainData, SelectedFeatures_in_RankImportanceOrder] = WrapperFeatureSelection(Matrix, need_keepin, FeaturesPredefined)
%% Implement sequentialfs Matlab function
% 
% Suitable features tested may be used to train model.
% See the following temp results

% HDU, Bocheng Wang 2018.10
%%

X = Matrix(:, 2:size(Matrix, 2));
y = Matrix(:, 1);

c = cvpartition(y,'KFold',5);
opts = statset('display', 'iter',  'TolTypeFun','rel', 'TolFun', 1e-16);

%% core features set that were selected from BCT analysis (P<0.1)

keepin_Index = FeaturesPredefined;

% multi-class recognition
fun = @(train_data,train_labels,test_data,test_labels) ...
    sum(predict(fitcecoc(train_data,train_labels), test_data) ~= test_labels);   
% end multi-class

% two-class recognition
% fun = @(train_data,train_labels,test_data,test_labels) ...
%   sum(predict(fitcsvm(train_data,train_labels,'KernelFunction','rbf'), test_data) ~= test_labels); 
% end two-class recognition
if need_keepin == true
    [fs,history] = sequentialfs(fun,X,y,'cv',c,'options',opts, 'keepin', keepin_Index);
else
    [fs,history] = sequentialfs(fun,X,y,'cv',c,'options',opts);
end
SelectedLabel = 1:size(Matrix, 2);
SelectedLabel = SelectedLabel(fs);

SelectedTrainData = [X(:, SelectedLabel), y];
SelectedFeatures_in_RankImportanceOrder = SelectedLabel;

%% temp results  HC_vs_EMCI 
% 
%
% Selected Features in Rank Importance Order          Matlab Machine Learning Toolbox
%
% [1 27 38 58 291 ];                                        Accuracy: 93.8%
% [1 24 27 73 264 633 1341 ];                               Accuracy: 95.8%
% [1 151 252 452 832 1075 1554];                            Accuracy: 93.8%  ¡Ì used in paper
% [1 73 401 422 654];                                       Accuracy: 91.7%
% [1 27 38 48 291 ];                                        Accuracy: 87.5%
% [1 73 87 401 633 750 ];                                   Accuracy: 87.5%
% [1 25 38 59 422 ];                                        Accuracy: 93.8%
% [1 4 73 87 174 633];                                      Accuracy: 85.4%
%% temp results  HC_vs_MCI
% 
% [2 3 470 575 626 784];                                    Accuracy: 91.7%  ¡Ì used in paper
% [3 64 96 250 357 598 ];                                   Accuracy: 87.5%
% [29 219 988 1050 ];                                       Accuracy: 89.6%
% [3 232 505 519 748 985 ];                                 Accuracy: 91.7%
% [3 7 297 1412 ];                                          Accuracy: 91.7%
% [3 131 173 186 313 598 627 ];                             Accuracy: 89.6%
% [3 56 116 341 646 980 ];                                  Accuracy: 91.7%
% [2 3 7 13 95 96 222];                                     Accuracy: 87.5%
%% temp results  HC_vs_LMCI
%
% [32 142 270 297 375];                                     Accuracy: 91.7%
% [11 58 140 149 1174];                                     Accuracy: 83.3%
% [4 9 23 33 87 175 176 1054 ];                             Accuracy: 95.8%
% [5 6 27 161 172 282 ];                                    Accuracy: 85.4%
% [40 51 58 67 143 337];                                    Accuracy: 87.5%
% [8 112 172 418 541 1014 ];                                Accuracy: 85.4%
% [5 6 112 167 172 197 239 1504 ];                          Accuracy: 95.8%  ¡Ì used in paper
% [5 172 191 355 492 1267 ];                                Accuracy: 89.6%
%% temp results  HC_vs_AD
%
% [83 208 223 238 350 772 ];                                Accuracy: 91.7%
% [21 249 308 548 1381];                                    Accuracy: 87.5%
% [83 238 384 558 652];                                     Accuracy: 93.8%
% [21 22 79 185 ];                                          Accuracy: 95.8%  ¡Ì used in paper
% [21 732 907 ];                                            Accuracy: 91.7%
% [21 79 185 862 ];                                         Accuracy: 91.7%
% [19 216 408 484 ];                                        Accuracy: 85.4%
% [21 79 185 862 ];                                         Accuracy: 89.6%
%% temp results  EMCI_vs_MCI
% 
% [10 20 120 160 169 343];                                  Accuracy: 87.5%
% [10 294 428 715 966 ];                                    Accuracy: 81.3%
% [10 409 487 504 892 1318 ];                               Accuracy: 91.7%  ¡Ì used in paper
% [10 504 1220 ];                                           Accuracy: 87.5%
% [10 296 343 449 504 779 969 ];                            Accuracy: 85.4%
% [14 68 169 343 ];                                         Accuracy: 89.6%
% [10 74 317 459 504 1017 1311 ];                           Accuracy: 85.4%
% [53 60 68 138 302 344 1195];                              Accuracy: 83.3%
% [10 276 316 475 504 1395 ];                               Accuracy: 83.3%
%% temp results  EMCI_vs_LMCI
% 
% [10 16 103 272 456 780 1180]                              Accuracy: 81.3%
% [108 275 496 1079 1159 ];                                 Accuracy: 81.3%
% [9 102 206 275 302 722 ];                                 Accuracy: 83.3%
% [10 234 396 546 692 ];                                    Accuracy: 87.5%  ¡Ì used in paper
% [10 39 234 563 1213 ];                                    Accuracy: 87.5%
% [45 63 275 519 1029 1268 ];                               Accuracy: 85.4%
% [10 12 39 64 245 261 540 1218 ];                          Accuracy: 85.4%
% [99 256 275 303 1618];                                    Accuracy: 81.3%
%% temp results  EMCI_vs_AD
% 
% [2 18 58 148 451 731 ];                                   Accuracy: 93.8%
% [9 19 22 350 ];                                           Accuracy: 91.7%
% [2 8 15 77 121 1157 ];                                    Accuracy: 95.8%
% [2 18 39 58 81 148 252 ];                                 Accuracy: 95.8%
% [2 6 70 487 1609];                                        Accuracy: 91.7%
% [2 8 77 121 ];                                            Accuracy: 91.7%
% [2 6 8 70 821];                                           Accuracy: 93.8%
% [9 22 620 653 986 ];                                      Accuracy: 91.7%
% [2 8 15 30 ];                                             Accuracy: 87.5%
%% temp results  MCI_vs_LMCI
% 
% [16 26 91 160 257 292 694 1121 ];                         Accuracy: 89.6%
% [33 37 76 119 365];                                       Accuracy: 89.6%
% [16 45 47 292 794 ];                                      Accuracy: 87.5%
% [18 24 47 96 112 219 1304];                               Accuracy: 87.5%
% [16 45 47 112 292 794 ];                                  Accuracy: 89.6%
% [3 84 89 224 247 668 ];                                   Accuracy: 93.8%
% [57 98 119 129 153 156 174 330];                          Accuracy: 97.9%  ¡Ì used in paper
% [57 168 353 401 611 794 905 ];                            Accuracy: 95.8%
% [5 84 310 ];                                              Accuracy: 91.7%
%% temp results  MCI_vs_AD
% 
% [3 275 639 656];                                          Accuracy: 87.5%
% [3 93 152 155 598 ];                                      Accuracy: 91.7%
% [3 33 94 101 155 843 ];                                   Accuracy: 97.9%
% [66 193 324 704 1255 ];                                   Accuracy: 91.7%
% [3 177 386 1372 ]                                         Accuracy: 91.7%
% [66 623 1077];                                            Accuracy: 87.5%
% [66 533 621 766 1018 1312 ];                              Accuracy: 89.6%
% [66 193 732 ];                                            Accuracy: 85.4%
% [66 121 193 326 818 ];                                    Accuracy: 93.8%
%% temp results  LMCI_vs_AD
% 
% [16 18 99 162 173 ];                                      Accuracy: 91.7%
% [27 78 498 1081 ];                                        Accuracy: 91.7%
% [27 78 388 967 ];                                         Accuracy: 91.7%
% [4 10 71 157 320];                                        Accuracy: 89.6%
% [4 8 54 71 204 1340 ];                                    Accuracy: 95.8%  ¡Ì used in paper
% [4 8 54 71 377 1340 ];                                    Accuracy: 87.5%
% [4 254 476 791 ];                                         Accuracy: 93.8%
% [4 109 765 1146];                                         Accuracy: 85.4%
% [4 41 130 249 313 440 ];                                  Accuracy: 95.8%
% [27 388 664 ];                                            Accuracy: 85.4%

%% temp result HC_vs_EMCI_LMCI_AD
% [8 295 419 724 743 881 1095 1272 1341 1543 ];             Accuracy: 50%
% [12 22 277 610 1439 1543 ];                               Accuracy: 53%

%% temp result HC_vs_MCI_vs_AD
% [2 7 10 11 17 37 41 56 75 81 116 142 152 172 173 178 181 189 201 247 460 473 499 512 570 588 612 640 742 759 810 858 969 1005 1059 1208 1322 1346 1356 1421 1489 1655 1665 1734 1772 1879 2082 2215 2297 2299 2330 2401 2460];
