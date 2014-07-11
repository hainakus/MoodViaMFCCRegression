% function get mfcc data from .mat files from Erik Schmidt
% these data exports in text file

names = []
data = []

files = dir('*.mat');

for i = 1:length(files)
    name = files(i).name
    load(name);
    
    % get mena of mfccs
    % return 20x1 matrix
    mfcc = mean(features.chroma, 2);    
    
    data = [data; mfcc'];
    size(data)
end
size(data)
fid = fopen('chroma.csv','w');
for i = 1:length(files)
    name = files(i).name;
    fprintf(fid,'%s',name);
    for j = data(i,:)
        fprintf(fid,',');
        fprintf(fid,'%f', j);
    end
   	fprintf(fid, '\n');
end
fclose(fid)