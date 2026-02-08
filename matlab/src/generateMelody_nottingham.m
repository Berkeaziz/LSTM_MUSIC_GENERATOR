% generateMelody_nottingham.m
clear; clc;

load("models/lstmModel_nottingham.mat","net","tokenVocabSize");
load("data/sequences_nottingham.mat", ...
    "allPitchSeqs","allDurClassSeqs","pitch2idx","idx2pitch", ...
    "pitchVocabSize","numDurClasses","durRep");

sequenceLength = 25;
genLength      = 80;

% seed: aynı melodinin başından al
iMel = randi(numel(allPitchSeqs));
seedP = allPitchSeqs{iMel};
seedD = allDurClassSeqs{iMel};

if numel(seedP) < sequenceLength
    error("Seçilen melodi çok kısa, tekrar dene.");
end

seedP = seedP(1:sequenceLength);
seedD = seedD(1:sequenceLength);

seedPIdx = arrayfun(@(p)pitch2idx(p), seedP);
seedTok  = seedPIdx + (seedD - 1) * pitchVocabSize;

temperature = 1.0;

genTok = seedTok;

for t = 1:genLength
    x = genTok(end-sequenceLength+1:end);
    x = x(:)';

    probs = predict(net, {x}, "ExecutionEnvironment","auto");
    probs = squeeze(probs);

    logits = log(probs + 1e-8) / temperature;
    probsT = exp(logits);
    probsT = probsT / sum(probsT);

    nextTok = randsample(1:tokenVocabSize, 1, true, probsT);
    genTok(end+1) = nextTok;
end

% token -> pitchIdx, durIdx
tokGenOnly = genTok(sequenceLength+1:end);

pitchIdx = mod(tokGenOnly - 1, pitchVocabSize) + 1;
durIdx   = floor((tokGenOnly - 1) / pitchVocabSize) + 1;

generatedPitches = arrayfun(@(ix) idx2pitch(ix), pitchIdx);

% duration temsil değeri (durRep) -> sentezde kullanılacak “ham süre”
durIdx = max(1, min(numel(durRep), durIdx));
generatedDur = durRep(durIdx);

fprintf("Üretilen melodi: %d nota.\n", numel(generatedPitches));

if ~isfolder("results")
    mkdir("results");
end

save(fullfile("results","generatedMelody_nottingham.mat"), ...
    "generatedPitches","generatedDur");
