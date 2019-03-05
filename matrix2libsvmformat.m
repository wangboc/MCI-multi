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
    fp=fopen(['.\' filename],'a');%'A.txt'为文件名；'a'为打开方式：在打开的文件末端添加数据，若文件不存在则创建。
    fprintf(fp,'%s \r\n',content);%fp为文件句柄，指定要写入数据的文件。注意：%d后有空格。  
    fclose(fp);%关闭文件。
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

        
       
