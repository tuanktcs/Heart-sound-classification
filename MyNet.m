
% lgraph = [
%     imageInputLayer(imageSize)
%     convolution2dLayer(5,20)
%     batchNormalizationLayer
%     reluLayer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];

lgraph = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     dropoutLayer(0.2)
    fullyConnectedLayer(100)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];