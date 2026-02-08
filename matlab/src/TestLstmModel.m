% TestLstmModel.m
% external\matlab-midi\tests\midi\jesu.mid dosyasını LSTM modeli ile test eder.
% Çıktılar:
%   results/external_midi_test_metrics_jesu.txt
%   results/ext_pitch_hist_jesu.png
%   results/ext_duration_hist_jesu.png
%   results/ext_note_time_jesu.png
%   results/ext_duration_class_jesu.png

clear; clc;

%% 0) matlab-midi TOOLBOX PATH (external\matlab-midi\src)
here = fileparts(mfilename("fullpath"));
root = here;

toolboxSrc = "";
for k = 1:6
    cand = fullfile(root, "external", "matlab-midi", "src");
    if isfolder(cand)
        toolboxSrc = cand;
        break;
    end
    root = fileparts(root);
end

assert(toolboxSrc ~= "", "external/matlab-midi/src bulunamadı. Klasör yapın farklı.");
addpath(genpath(toolboxSrc));
rehash toolboxcache;

assert(exist("readmidi","file")==2, "readmidi bulunamadı. toolboxSrc=%s", toolboxSrc);
assert(exist("midiInfo","file")==2,  "midiInfo bulunamadı. toolboxSrc=%s", toolboxSrc);

%% 1) Dosya yolları
seqPath   = fullfile("data","sequences_nottingham.mat");
dataPath  = fullfile("data","sequences_dataset_nottingham.mat");
modelPath = fullfile("models","lstmModel_nottingham.mat");

% Test MIDI (SENDE BÖYLE)
midiPath  = fullfile("external","matlab-midi","tests","midi","jesu.mid");

outDir = "results";
if ~isfolder(outDir), mkdir(outDir); end

assert(isfile(midiPath),  "Test MIDI yok: %s", midiPath);
assert(isfile(modelPath), "Model yok: %s", modelPath);
assert(isfile(seqPath),   "sequences_nottingham.mat yok: %s", seqPath);

[~, midiName, ~] = fileparts(midiPath);

%% 2) Model + mapping yükle
S = load(modelPath, "net");
net = S.net;

Z = load(seqPath, "pitch2idx","pitchVocabSize","numDurClasses","durEdges");
pitch2idx      = Z.pitch2idx;
pitchVocabSize = Z.pitchVocabSize;
numDurClasses  = Z.numDurClasses;
durEdges       = Z.durEdges;

tokenVocabSize = pitchVocabSize * numDurClasses;

%% 3) windowLen (dataset'ten)
windowLen = 25; % fallback
if isfile(dataPath)
    D = load(dataPath, "sequenceLength");
    if isfield(D,"sequenceLength") && ~isempty(D.sequenceLength)
        windowLen = D.sequenceLength;
    end
end
fprintf("Test windowLen = %d\n", windowLen);

%% 4) MIDI -> pitch & duration
[pSeq, dSeq] = midiToPitchSeq(midiPath);
assert(~isempty(pSeq) && ~isempty(dSeq), "MIDI okunamadı/boş: %s", midiPath);

%% 4.5) PNG çıktıları (pitch/duration/nota-zaman)
% 1) Pitch histogram
fig = figure('Visible','off');
histogram(pSeq, 60);
xlabel("MIDI Pitch"); ylabel("Count");
title("External MIDI Pitch Distribution");
grid on;
safeExport(fig, fullfile(outDir, "ext_pitch_hist_" + midiName + ".png"));
close(fig);

% 2) Duration histogram (ham)
fig = figure('Visible','off');
histogram(dSeq, 60);
xlabel("Duration (sec or time units)"); ylabel("Count");
title("External MIDI Duration Distribution");
grid on;
safeExport(fig, fullfile(outDir, "ext_duration_hist_" + midiName + ".png"));
close(fig);

% 3) Nota-zaman grafiği
t0 = [0, cumsum(dSeq(1:end-1))];
t1 = t0 + dSeq;

fig = figure('Visible','off'); hold on;
for i = 1:numel(pSeq)
    plot([t0(i) t1(i)], [pSeq(i) pSeq(i)], "LineWidth", 2);
end
xlabel("Time"); ylabel("MIDI Pitch");
title("External MIDI Note-Time Plot");
grid on; hold off;
safeExport(fig, fullfile(outDir, "ext_note_time_" + midiName + ".png"));
close(fig);

%% 5) duration -> class + class grafiği
dClass = discretize(dSeq, durEdges);

fig = figure('Visible','off');
edges = 0.5:1:(numDurClasses+0.5);
histogram(dClass, edges);
xlim([1 numDurClasses]);
xlabel("Duration Class"); ylabel("Count");
title("External MIDI Duration Class Distribution");
grid on;
safeExport(fig, fullfile(outDir, "ext_duration_class_" + midiName + ".png"));
close(fig);

%% 6) pitch -> idx (bilinmeyenleri ele), NaN class'ları ele
pIdx = zeros(size(pSeq));
okP = true(size(pSeq));
for i = 1:numel(pSeq)
    if isKey(pitch2idx, pSeq(i))
        pIdx(i) = pitch2idx(pSeq(i));
    else
        okP(i) = false;
    end
end

okD = ~isnan(dClass);
ok  = okP & okD;

pIdx   = pIdx(ok);
dClass = dClass(ok);

tokens = pIdx + (double(dClass)-1) * pitchVocabSize;

if numel(tokens) <= windowLen
    error("Test MIDI çok kısa: token sayısı=%d, windowLen=%d", numel(tokens), windowLen);
end

%% 7) Windows üret + metrik
[Xtest, YtestNum] = buildWindowsFromTokens(tokens, windowLen);
Ytest = categorical(YtestNum(:), 1:tokenVocabSize);

[acc, nll, ppl] = evalNextTokenMetrics(net, Xtest, Ytest);

%% 8) Kaydet
outPath = fullfile(outDir, "external_midi_test_metrics_" + midiName + ".txt");

fid = fopen(outPath,"w");
fprintf(fid, "=== EXTERNAL MIDI TEST ===\n");
fprintf(fid, "File: %s\n", midiPath);
fprintf(fid, "Saved at: %s\n\n", datestr(now));
fprintf(fid, "windowLen: %d\n", windowLen);
fprintf(fid, "Num windows: %d\n", numel(Xtest));
fprintf(fid, "Accuracy (top-1): %.4f\n", acc);
fprintf(fid, "Avg NLL: %.6f\n", nll);
fprintf(fid, "Perplexity: %.4f\n", ppl);
fclose(fid);

fprintf("Bitti.\nTXT: %s\nPNG: results/ext_*_%s.png\n", outPath, midiName);

%% ---------------- LOCAL FUNCTIONS ----------------
function [X, Y] = buildWindowsFromTokens(tokens, windowLen)
    n = numel(tokens) - windowLen;
    X = cell(1, n);
    Y = zeros(1, n);

    for t = 1:n
        X{t} = tokens(t:t+windowLen-1);
        Y(t) = tokens(t+windowLen);
    end
end

function [acc, nll, ppl] = evalNextTokenMetrics(net, X, Ycat)
    n = numel(X);
    if n == 0
        acc = NaN; nll = NaN; ppl = NaN; return;
    end

    correct = 0;
    nllSum  = 0;

    for i = 1:n
        scores = predict(net, X{i});
        p = scores(:)';

        yTrue = double(Ycat(i));
        [~, yHat] = max(p);

        if yHat == yTrue
            correct = correct + 1;
        end

        pTrue = max(1e-12, p(yTrue));
        nllSum = nllSum - log(pTrue);
    end

    acc = correct / n;
    nll = nllSum / n;
    ppl = exp(nll);
end

function safeExport(fig, outPath)
    % exportgraphics yoksa saveas'e düş
    try
        exportgraphics(fig, outPath);
    catch
        try
            saveas(fig, outPath);
        catch
        end
    end
end
