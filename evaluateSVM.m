function output = evaluateSVM(filename)
%% Invoke libSVM tools. To achieve this, python should be installed.
% HDU, Bocheng Wang, 2018.10
path = pwd;
cmd  = ['python ' path '\easy.py ' path '\' filename];
output = system(cmd);

