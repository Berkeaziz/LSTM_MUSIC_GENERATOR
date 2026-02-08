% prepare_nottingham_sequences.m
% Amaç: data/nottingham_midi içindeki tüm .mid dosyalarından pitch + duration dizileri çıkar,
% duration'ı numDurClasses sınıfa böl, allPitchSeqs + allDurClassSeqs kaydet.

clear; clc;

midiDir = fullfile("data","nottingham_midi");
files   = dir(fullfile(midiDir, "*.mid"));

if isempty(files)
    error("data/nottingham_midi içinde .mid dosyası bulunamadı.");
end

minLen = 16;

% Duration sınıf sayısı (ritim çözünürlüğün)
numDurClasses = 32;

allPitchSeqs = {};
allDurSeqs   = {};
allDurAll    = [];   % tüm dosyalardan duration havuzu (binleme için)

for k = 1:numel(files)
    midiPath = fullfile(files(k).folder, files(k).name);

    [pitchSeq, durSeq] = midiToPitchSeq(midiPath);

    if numel(pitchSeq) < minLen
        continue;
    end
    if numel(durSeq) ~= numel(pitchSeq)
        continue;
    end

    allPitchSeqs{end+1} = pitchSeq; %#ok<SAGROW>
    allDurSeqs{end+1}   = durSeq;   %#ok<SAGROW>

    allDurAll = [allDurAll, durSeq]; %#ok<AGROW>
end

fprintf("Toplam %d MIDI dosyasından %d melodi okundu.\n", ...
    numel(files), numel(allPitchSeqs));

if isempty(allPitchSeqs)
    error("Hiç yeterince uzun sekans bulunamadı. minLen değerini düşürmeyi deneyebilirsin.");
end

% --- Pitch vocab / mapping ---
allNotes  = [allPitchSeqs{:}];
vocab     = unique(allNotes);
pitchVocabSize = numel(vocab);

pitch2idx = containers.Map('KeyType','double','ValueType','double');
idx2pitch = containers.Map('KeyType','double','ValueType','double');

for i = 1:pitchVocabSize
    p = vocab(i);
    pitch2idx(p) = i;
    idx2pitch(i) = p;
end

% --- Duration binleme: edges + temsil süre (repDur) üret ---
[durEdges, durRep] = makeDurationBins(allDurAll, numDurClasses);

% --- Her melodiyi duration class dizisine çevir ---
allDurClassSeqs = cell(size(allDurSeqs));
for i = 1:numel(allDurSeqs)
    d = allDurSeqs{i};
    durClass = discretize(d, durEdges);  % 1..K
    allDurClassSeqs{i} = durClass;
end

if ~isfolder("data")
    mkdir("data");
end

save(fullfile("data","sequences_nottingham.mat"), ...
    "allPitchSeqs","allDurSeqs","allDurClassSeqs", ...
    "vocab","pitchVocabSize","pitch2idx","idx2pitch", ...
    "numDurClasses","durEdges","durRep");

fprintf("Nottingham dataset hazır: %d melodi, pitchVocabSize=%d, numDurClasses=%d\n", ...
    numel(allPitchSeqs), pitchVocabSize, numDurClasses);

% ---------------- Local function ----------------
function [edges, repDur] = makeDurationBins(durs, K)
    durs = durs(:);
    durs = durs(isfinite(durs) & durs > 0);
    if isempty(durs)
        error("Duration havuzu boş/bozuk. MIDI okuma tarafında problem olabilir.");
    end

    d = sort(durs);
    n = numel(d);

    % K sınıf için K+1 edge gerekir. Uçları -Inf / Inf yapıyoruz.
    p = linspace(0, 1, K+1);
    idx = round(p*(n-1) + 1);
    idx = max(1, min(n, idx));

    edges = d(idx);
    edges(1)   = -Inf;
    edges(end) =  Inf;

    % monotonik yap (duplicate edge olursa discretize sıkıntı çıkarabilir)
    for i = 2:numel(edges)-1
        if edges(i) <= edges(i-1)
            edges(i) = edges(i-1) + eps(edges(i-1) + 1);
        end
    end

    cls = discretize(durs, edges);
    Kreal = max(cls);

    repDur = zeros(1, Kreal);
    for c = 1:Kreal
        vals = durs(cls == c);
        if isempty(vals)
            repDur(c) = NaN;
        else
            repDur(c) = median(vals);
        end
    end

    % boş sınıf kaldıysa komşudan doldur
    for c = 1:numel(repDur)
        if isnan(repDur(c))
            left = find(~isnan(repDur(1:c-1)), 1, "last");
            right = find(~isnan(repDur(c+1:end)), 1, "first");
            if isempty(left) && isempty(right)
                repDur(c) = median(durs);
            elseif isempty(left)
                repDur(c) = repDur(c + right);
            elseif isempty(right)
                repDur(c) = repDur(left);
            else
                repDur(c) = 0.5*(repDur(left) + repDur(c + right));
            end
        end
    end
end
