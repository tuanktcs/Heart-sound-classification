function [outdata,outlabel]=segsig(x,labels,stereo)

[data,fs]=audioread(x);
% windowWidth = 27;
% polynomialOrder = 5;
% data = sgolayfilt(data, polynomialOrder, windowWidth);

%% Resampling into 44100 Hz
% fs_new = 2000;
% [Numer, Denom] = rat(fs_new/fs);
% data_new = resample(data, Numer, Denom);
% data = data_new;
% 
% fs = fs_new;
%% segment duration = 1s
SegmentDuration =0.5;
if (SegmentDuration*fs)/length(data)>1
    aa = ceil((SegmentDuration*fs)/length(data));
    data1 = data;
    for j=2:aa
        data=[data1;data];
    end
else
    data = data;
end
% data=bandpass(data,[50 1000],fs);


fs=fs*SegmentDuration;
        out(:,1)=data(1:fs);
        label{1}=labels;
    outdata=[out(:,:)];
outlabel=[label(:)];

