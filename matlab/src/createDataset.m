%% createDataset.m
% Nottingham pitch+durationClass sekanslarından token dataset oluşturur.

clear; clc;

load("data/sequences_nottingham.mat", ...
    "allPitchSeqs","allDurClassSeqs","pitch2idx", ...
    "pitchVocabSize","numDurClasses");

sequenceLength = 25;
numSequences   = numel(allPitchSeqs);

tokenVocabSize = pitchVocabSize * numDurClasses;

XTrain = {};
XVal   = {};
YTrain_idx = [];
YVal_idx   = [];

valRatio = 0.2;

for i = 1:numSequences
    pSeq = allPitchSeqs{i};
    dCls = allDurClassSeqs{i};

    if numel(pSeq) <= sequenceLength || numel(dCls) ~= numel(pSeq)
        continue;
    end

    % pitch -> idx
    pIdx = arrayfun(@(p) pitch2idx(p), pSeq);

    % token = pIdx + (dCls-1)*pitchVocabSize
    tok = pIdx + (dCls - 1) * pitchVocabSize;

    for t = 1:(length(tok) - sequenceLength)
        x = tok(t:t+sequenceLength-1);
        y = tok(t+sequenceLength);

        if rand < valRatio
            XVal{end+1,1} = x; %#ok<SAGROW>
            YVal_idx(end+1) = y;
        else
            XTrain{end+1,1} = x; %#ok<SAGROW>
            YTrain_idx(end+1) = y;
        end
    end
end

YTrain = categorical(YTrain_idx, 1:tokenVocabSize);
YVal   = categorical(YVal_idx,   1:tokenVocabSize);

fprintf("Dataset hazır: %d train, %d val örnek. tokenVocabSize=%d\n", ...
    numel(XTrain), numel(XVal), tokenVocabSize);

save("data/sequences_dataset_nottingham.mat", ...
    "XTrain","YTrain","XVal","YVal", ...
    "tokenVocabSize","sequenceLength", ...
    "pitchVocabSize","numDurClasses");
