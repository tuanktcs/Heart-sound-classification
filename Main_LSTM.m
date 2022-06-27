%% Automatic detection sound quality of guitar using Deep Learning
% This program is used to train a deep learning model that detects
% quality of a guitar through sound.
%% Load audio Data Set
clear; clc;
addpath(pwd,'Spectrogram')
addpath(pwd,'Networks')
datafolder = fullfile(pwd,'Proprocessed_Data_2s');
ads = audioDatastore(datafolder, ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')
ads0 = copy(ads);


Lables = categorical(unique(ads.Labels))';
countEachLabel(ads)


%% Split Data into Training and Test Sets
numTrainFiles=0.7; % 80% for the training dataset
[adsTrain,adsTest] = splitEachLabel(ads,numTrainFiles,'randomize');

%% Compute Speech Spectrograms
% To prepare the data for efficient training of a convolutional neural
% network, convert the raw waveforms to log-mel spectrograms.
%
% Define the parameters of the spectrogram calculation. |segmentDuration|
% is the duration of each audio clip (in seconds). |frameDuration| is the
% duration of each frame for spectrogram calculation. |hopDuration| is the
% time step between each column of the spectrogram. |numBands| is the
% number of log-mel filters and equals the height of each spectrogram.
segmentDuration = 2;
frameDuration = 0.025;
hopDuration = 0.01;
numBands = 14;
fs = 8000;
%%
% Compute the spectrograms for the training and test sets
% by using the supporting function
% |AudioSpectrograms|>. The |AudioSpectrograms| function uses
% |designAuditoryFilterBank| for the log-mel spectrogram calculations. To obtain data
% with a smoother distribution, take the logarithm of the spectrograms
% using a small offset |epsil|.
epsil = 1e-6;

XTrain = AudioSpectrograms(adsTrain,segmentDuration,frameDuration,hopDuration,numBands);
% XTrain = log10(XTrain + epsil);

XTest = AudioSpectrograms(adsTest,segmentDuration,frameDuration,hopDuration,numBands);
% XTest = log10(XTest + epsil);

YTrain = adsTrain.Labels;
YTest = adsTest.Labels;

%% Visualize Data
% Plot the waveforms and spectrograms of a few training examples. Play the
% corresponding audio clips.
specMin = min(XTrain(:));
specMax = max(XTrain(:));
% idx = randperm(size(XTrain,4),5);
idx(1)= 60; idx(2) = 220; idx(3) = 360; idx(4) = 550; idx(5) = 700;
figure(1);
t = linspace(0,segmentDuration,segmentDuration*fs);
for i = 1:5
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    subplot(3,5,i)
    plot(t,x(1:(end)))
    xlabel('Time (s)')
    axis tight
    title(string(adsTrain.Labels(idx(i))))
    ylim([-0.6 0.6])
    subplot(3,5,i+5)
    spect = XTrain(:,:,1,idx(i));
    pcolor(spect)
    caxis([specMin+2 specMax])
    shading flat
    title('Spectrogram');
    subplot(3,5,i+10)
    MFCC = mfcc(x,fs);
    for j=1:14
    hold on
    plot(MFCC(j,:))
    end
    ylim([-25 10])
    title('MFCC');
    sound(x,fs)
%     pause(2)
end

%%
% Training neural networks is easiest when the inputs to the network have a reasonably
% smooth distribution and are normalized. To check that the data distribution
% is smooth, plot a histogram of the pixel values of the training data.
figure(2);
histogram(XTrain,'EdgeColor','none','Normalization','pdf')
axis tight
ax = gca;
ax.YScale = 'log';
xlabel("Input Pixel Value")
ylabel("Probability Density")


%%
% Plot the distribution of the different class labels in the training and
% validation sets. The test set has a very similar distribution to the
% validation set.
figure(3);
subplot(2,1,1)
histogram(YTrain)
title("Training Label Distribution")
subplot(2,1,2)
histogram(YTest)
title("Test Label Distribution")

%% Add Data Augmentation
% Create an augmented image datastore for automatic augmentation and
% resizing of the spectrograms. Translate the spectrogram randomly up to 10
% frames (600 ms) forwards or backwards in time, and scale the spectrograms
% along the time axis up or down by 10 percent. Augmenting the data can
% increase the effective size of the training data and help prevent the
% network from overfitting. The augmented image datastore creates augmented
% images in real time during training and inputs them to the network. No
% augmented spectrograms are saved in memory.
sz = size(XTrain);
sz2 = size(XTest);
specSize = sz(1:2);
imageSize = [specSize 1];


%% Define Neural Network Architecture
classWeights = 1./countcats(YTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(categories(YTrain));


for i=1:sz(4)
    KTrain{i}=XTrain(:,:,1,i)';
end
   KTrain = KTrain';
   for i=1:sz2(4)
       KTest{i} = XTest(:,:,1,i)';
   end
   KTest = KTest';
%% Design network
% augmenter = imageDataAugmenter( ...
%     'RandXTranslation',[-10 10], ...
%     'RandXScale',[0.9 1.1], ...
%     'FillValue',log10(epsil));
% augimdsTrain = augmentedImageDatastore(imageSize,KTrain,YTrain, ...
%     'DataAugmentation',augmenter);
% inputSize = specSize(1);
inputSize = imageSize(2);
numHiddenUnits = 100;
lgraph = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    lstmLayer(numHiddenUnits,'OutputMode','last')
%     dropoutLayer(0.2)
%     fullyConnectedLayer(100)
    fullyConnectedLayer(100)
    fullyConnectedLayer(50)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
%% Train Network
% Specify the training options. Use the Adam optimizer with initial 
% learning rate of 1e-5, a mini-batch size of 200, max epochs of 200.
options = trainingOptions('adam', ...
    'InitialLearnRate',0.000002, ...
    'MaxEpochs',2000, ...
    'MiniBatchSize',75,...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment',"gpu",...
    'ValidationData',{KTest,YTest}, ...
    'ValidationFrequency',1000, ...
    'Verbose',false, ...
    'Plots','training-progress');
% Train the network. If you do not have a GPU, then training the network
% can take time.
    trainedNet = trainNetwork(KTrain,YTrain,lgraph,options);

%% Evaluate Trained Network
% Calculate the final accuracy of the network on the training set (without
% data augmentation) and validation set. The network is very accurate on
% this data set. However, the training and test data all have
% similar distributions that do not necessarily reflect real-world
% environments.
YValPred = classify(trainedNet,KTest);
validationError = mean(YValPred ~= YTest);
YTrainPred = classify(trainedNet,KTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Training error: " + trainError*100 + "%")
disp("Validation error: " + validationError*100 + "%")

%%
% Plot the confusion matrix. Display the precision and recall for each
% class by using column and row summaries. Sort the classes of the
% confusion matrix.
figure(4);
cm = confusionchart(YTest,YValPred);
% cm.Title = 'Confusion Matrix for Validation Data';
% cm.ColumnSummary = 'column-normalized';
% cm.RowSummary = 'row-normalized';
info = whos('trainedNet');
disp("Network size: " + info.bytes/1024 + " kB")

for i=1:100
    x = randn(inputSize);
    tic
    [YPredicted,probs] = classify(trainedNet,x,"ExecutionEnvironment",'gpu');
    time(i) = toc;
end
disp("Single-image prediction time on GPU: " + mean(time(11:end))*1000 + " ms")


