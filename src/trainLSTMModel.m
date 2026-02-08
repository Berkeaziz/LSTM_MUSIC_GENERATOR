% trainLSTMModel.m
clear; clc;

% ====== TRAIN KİLİDİ (model varsa tekrar eğitme) ======
modelPath = fullfile("models","lstmModel_nottingham.mat");

forceTrain = false;   % true yaparsan model olsa bile yeniden train eder

if isfile(modelPath) && ~forceTrain
    load(modelPath, "net", "tokenVocabSize");
    fprintf("Model zaten var: %s\nTraining atlandi.\n", modelPath);
    return;
end
% ======================================================

load("data/sequences_dataset_nottingham.mat", ...
    "XTrain","YTrain","XVal","YVal","tokenVocabSize");

inputSize      = 1;
embeddingDim   = 64;
numHiddenUnits = 256;

layers = [
    sequenceInputLayer(inputSize, "Name","input")
    wordEmbeddingLayer(embeddingDim, tokenVocabSize, "Name","embed")
    lstmLayer(numHiddenUnits, "OutputMode","last", "Name","lstm")
    fullyConnectedLayer(tokenVocabSize, "Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","output")
];

miniBatchSize = 128;
maxEpochs     = 20;

options = trainingOptions("adam", ...
    "InitialLearnRate", 1e-3, ...
    "MaxEpochs", maxEpochs, ...
    "MiniBatchSize", miniBatchSize, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", {XVal, YVal}, ...
    "ValidationFrequency", 200, ...
    "ExecutionEnvironment","gpu", ...
    "Plots", "training-progress", ...
    "Verbose", true);

net = trainNetwork(XTrain, YTrain, layers, options);

if ~isfolder("models")
    mkdir("models");
end

save(modelPath, "net", "tokenVocabSize");
fprintf("Model egitildi ve kaydedildi: %s\n", modelPath);
