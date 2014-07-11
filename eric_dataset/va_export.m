% load and read va from file

load('msLiteTurk.mat')

valence = msLiteTurk.valence
arousal = msLiteTurk.arousal
idx = msLiteTurk.songid

fida = fopen('valence.csv','w');
fidv = fopen('arousal.csv','w');
for i = 1:length(idx)
    i
    id = idx(i)
    fprintf(fida,'%d',id{1});
    fprintf(fidv,'%d',id{1});
    val = valence(i);
    aro = arousal(i);
    for j = 1:length(val{1})
        v = val{1};
        a = aro{1};
        fprintf(fidv,',');
        fprintf(fidv,'%f', v(j)/200);
        fprintf(fida,',');
        fprintf(fida,'%f', a(j)/200);
    end
   	fprintf(fidv, '\n');
    fprintf(fida, '\n');
end

fclose(fidv);
fclose(fida);