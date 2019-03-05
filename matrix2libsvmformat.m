function [] = matrix2libsvmformat (matrix, filename)
%% convert matlab matrix into the format of libSVM required.

if exist(['.\' filename], 'file')
    delete(['.\' filename]);
end
for row_index = 1:size(matrix, 1)
    row = matrix(row_index, :);
    type = num2str(row(size(row, 2)));
    content = [type ' '];
    features = row(1:size(row, 2) - 1);
    for column_index = 1:size(features, 2)
        item = [num2str(column_index) ':' num2str(features(column_index)) ' '];
        content = [content item];
    end
    fp=fopen(['.\' filename],'a');%'A.txt'Ϊ�ļ�����'a'Ϊ�򿪷�ʽ���ڴ򿪵��ļ�ĩ��������ݣ����ļ��������򴴽���
    fprintf(fp,'%s \r\n',content);%fpΪ�ļ������ָ��Ҫд�����ݵ��ļ���ע�⣺%d���пո�  
    fclose(fp);%�ر��ļ���
end



% parselibSVM(cmdout);

% label = matrix(:, 1);
% features = matrix(:, 2:size(matrix, 2));
% 
% 
% model  = libsvmtrain(label, features, '-v 5');
% % [predict_label, accuracy, dec_values] = svmpredict(label, features, model) % test the training data
% 
% 

        
       
