% function get mfcc data from .mat files from Erik Schmidt
% these data exports in text file

names = []
datas = []

files = dir('*.mat');

for i = 1:length(files)
    name = files(i).name
    load(name)  %C:/Users/Primoz/Desktop/mslite/msLiteFeatures/'
    
    % get mena of mfccs
    % return 20x1 matrix
    mfcc = mean(features.ceps, 2);
    
    
    datas = [datas; mfcc'];
end

fid = fopen('mfccs','w');
for i = 1:length(files)
    name = files(i).name;
    fprintf(fid,'%s ',name);
    fprintf(fid, '%.10f ', datas(i,:));
   	fprintf(fid, '\n');
end
fclose(fid)