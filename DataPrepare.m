tic;
clear;
addpath(genpath(pwd));

% classes = {  '4.AD';'0.HC'; '1.EMCI'; '2.MCI'; '3.LMCI';};
classes = {  '0.HC'; '1.EMCI'; };
% classes = {'3.LMCI';};
for index = 1:size(classes)
    path = ['./Data_with_HC=24/Connectivity/'  char(classes(index, 1)) '/'];
    files = dir(path);
    subjects = zeros(24, 5777);
    subject_index = 1;
    for file_index = 1:size(files, 1)
        if files(file_index).isdir == 0
            load([path files(file_index).name]);
            [        global_efficiency_wei, ...
                     maximized_modularity_wei, ...
                     assortativity_wei, ...
                     optimal_number_of_modules_wei, ...
                     small_wordness_index_wei, ...
                     characteristic_path_length_wei, ...
                     mean_clustering_coefficient_wei,  ...
                     ...
                     global_efficiency_bin, ...
                     maximized_modularity_bin, ...
                     assortativity_bin, ...
                     optimal_number_of_modules_bin, ...
                     small_wordness_index_bin, ...
                     characteristic_path_length_bin, ...
                     mean_clustering_coefficient_bin, ...
                     ...
                     strength_wei, ...
                     clustering_coef_wei, ...
                     local_efficiency_wei, ...
                     betweenness_wei, ...
                     eigenvector_wei, ...
                     pagerank_wei, ...
                     degree_wei, ...
                     ...
                     strength_bin, ...
                     clustering_coef_bin, ...
                     local_efficiency_bin, ...
                     betweenness_bin, ...
                     eigenvector_bin, ...
                     subgraph_bin, ...
                     pagerank_bin, ...
                     kcoreness_bin, ...
                     flow_coefficiency, ...  
                     ...
                     PSW_optimal_wei, ...
                     PSW_optimal_bin ...
            ] = BCT_calculation(result);
%% feature �ṹ�� 7 weighted global metrics + 7 binary global metrics + 7 * 360 weighted local metrics + 9 * 360 binary local metrics 
            features = [     global_efficiency_wei, ...
                             maximized_modularity_wei, ...
                             assortativity_wei, ...
                             optimal_number_of_modules_wei, ...
                             small_wordness_index_wei, ...
                             characteristic_path_length_wei, ...
                             mean_clustering_coefficient_wei,  ...
                             ...
                             global_efficiency_bin, ...
                             maximized_modularity_bin, ...
                             assortativity_bin, ...
                             optimal_number_of_modules_bin, ...
                             small_wordness_index_bin, ...
                             characteristic_path_length_bin, ...
                             mean_clustering_coefficient_bin, ...
                             ...
                             strength_wei, ...
                             clustering_coef_wei, ...
                             local_efficiency_wei, ...
                             betweenness_wei, ...
                             eigenvector_wei, ...
                             pagerank_wei, ...
                             degree_wei, ...
                             ...
                             strength_bin, ...
                             clustering_coef_bin, ...
                             local_efficiency_bin, ...
                             betweenness_bin, ...
                             eigenvector_bin, ...
                             subgraph_bin, ...
                             pagerank_bin, ...
                             kcoreness_bin, ...
                             flow_coefficiency, ... 
                             ...
                             PSW_optimal_wei, ...
                             PSW_optimal_bin ...
                  ];
            type = char(classes(index, 1));
            if  type(1) == '0'
                type = 1;
            elseif type(1) == '1'
                type = 10;
            elseif type(1) == '2'
                type = 100;
            elseif type(1) == '3'
                type = 1000;
            elseif type(1) == '4'
                type = 10000;
            end
            subject = [type, features];
            subjects(subject_index, :) = subject;
            subject_index = subject_index + 1;
        end
    end
    name = ['./Data_with_HC=24/BCTs/' char(classes(index, 1)) '.mat'];
    save(name, 'subjects');      
end
toc;