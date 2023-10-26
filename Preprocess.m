%data->N data
%% Read data and pre-segmant
clear;
addpath(pwd,'stft')
addpath(pwd,'Signal_Toolbox/')

folder=fullfile(pwd,'Original Data')
ADS = audioDatastore(folder,'IncludeSubfolders',true,...
                            'LabelSource' ,'foldernames',...
                            'FileExtensions' ,'.wav' );
                       
[outdata,outlabel]=segsig(ADS.Files{1},char(ADS.Labels(1)),0);
N=length(ADS.Files);
for n=2:N
    [outdata2,outlabel2]=segsig(ADS.Files{n},char(ADS.Labels(n)),0);
    outdata=[outdata outdata2];
    outlabel=[outlabel;outlabel2];
    disp(['index:',num2str(n)]);
end

%% Train database (RAW)
Xtrain=outdata';
Ytrain=outlabel;

%% Feature extraction
R=1024;
sel=1;
fs=8000;
N=length(outlabel);
  pathst= fullfile(pwd,'Proprocessed_Data_0.5s');
    if exist(pathst, 'dir')
        rmdir(pathst,'s');
    end

sname = {'N';'MVP';'MS';'MR';'AS'};
for i=1:length(sname)
    pathst= fullfile(pwd,fullfile('Proprocessed_Data_0.5s',sname{i}));
    if ~exist(pathst, 'dir')
        mkdir(pathst)
    end
end
for i=1:N
    pathst=fullfile(pwd,fullfile('Proprocessed_Data_0.5s',outlabel{i}));
x=Xtrain(i,:)';
x=ssubmmsev(x,fs);
% audiowrite(fullfile(pathst,[outlabel{i},'-',num2str(i),'.wav']),x,fs);
audiowrite(fullfile(pathst,[char(ADS.Files{i,1}(end-13:end))]),x,fs);
% s
end


