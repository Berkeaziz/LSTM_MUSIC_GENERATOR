% evaluateModel_nottingham.m
% - Training pitch dağılımı vs generated pitch dağılımı
% - Kanıt paketi (tek dosyada):
%   layer kanıtı, duration dağılımı, nota-zaman, split sayıları,
%   song-level test metrikleri, window length ablation

clear; clc;

seqPath = fullfile("data","sequences_nottingham.mat");
genPath = fullfile("results","generatedMelody_nottingham.mat");

if ~isfile(seqPath) || ~isfile(genPath)
    error("Gerekli dosyalar yok. prepare_nottingham_sequences + generateMelody calistir.");
end

load(seqPath, "allPitchSeqs");
G = load(genPath);

trainP = cell2mat(allPitchSeqs);

% --- generated pitch alanı: iki isimden birini kabul et ---
if isfield(G,"pitchSeq")
    genP = G.pitchSeq;
elseif isfield(G,"generatedPitches")
    genP = G.generatedPitches;
else
    error("generatedMelody_nottingham.mat icinde pitchSeq veya generatedPitches yok.");
end

figure;
subplot(1,2,1);
histogram(trainP);
title("Training Pitch Distribution");

subplot(1,2,2);
histogram(genP);
title("Generated Pitch Distribution");

% Perplexity (opsiyonel): training-progress'ten son val loss'u yazarsin
valLoss = 0.0;
if valLoss > 0
    ppl0 = exp(valLoss);
    fprintf("Val Perplexity ~ %.3f\n", ppl0);
else
    fprintf("Perplexity icin valLoss gir (evaluateModel_nottingham.m).\n");
end


%% ========================================================================
%  KANIT PAKETİ (TEK DOSYA) — İSTERLERİN HEPSİ BURADA
%  Çıktılar: results/ klasörüne .png ve .txt olarak yazılır.
%% ========================================================================

outDir = "results";
if ~isfolder(outDir), mkdir(outDir); end

rng(42); % split/ablation tekrar üretilebilir olsun

% ---- Ayarlar ----
doSongSplitTrainAndTest = true;   % test datası performansı (mail-1)
doWindowLengthAblation  = true;   % windowLen ablation (mail-2)

songSplit_windowLen     = 25;     % raporda kullandığın windowLen
songSplit_maxEpochs     = 10;     % 20 daha iyi ama uzun
songSplit_miniBatch     = 128;

maxWindowsPerSet = 60000;         % RAM için limit (0: limit yok)

ablation_windowLens = [8 16 32 64];
ablation_epochs     = 3;
ablation_miniBatch  = 128;

% -------------------------------------------------------------------------
% 0) Pitch dağılımı figürünü dosyaya kaydet
% -------------------------------------------------------------------------
try
    exportgraphics(gcf, fullfile(outDir,"pitch_distribution_train_vs_generated.png"));
catch
end

% -------------------------------------------------------------------------
% 1) LAYER KANITI (model_layers.txt + model_layers.png)
% -------------------------------------------------------------------------
modelPath = fullfile("models","lstmModel_nottingham.mat");
if isfile(modelPath)
    S = load(modelPath, "net");
    net = S.net;

    % TXT
    txtPath = fullfile(outDir, "model_layers.txt");
    fid = fopen(txtPath, "w");
    fprintf(fid, "=== MODEL LAYERS EVIDENCE ===\n");
    fprintf(fid, "Saved at: %s\n\n", datestr(now));
    try
        L = net.Layers;
        fprintf(fid, "NumLayers: %d\n\n", numel(L));
        for i = 1:numel(L)
            fprintf(fid, "[%02d] %s (%s)\n", i, L(i).Name, class(L(i)));
        end
    catch ME
        fprintf(fid, "Layer okunamadi: %s\n", ME.message);
    end
    fclose(fid);

    % PNG
    pngPath = fullfile(outDir, "model_layers.png");
    try
        fig = figure('Visible','off');
        try
            plot(layerGraph(net.Layers));
            title("Model Layers (layerGraph)");
        catch
            axis off;
            text(0,0.7,"layerGraph plot edilemedi.","FontSize",12);
            text(0,0.6,"TXT kaniti kullan: results/model_layers.txt","FontSize",12);
        end
        exportgraphics(fig, pngPath);
        close(fig);
    catch
    end
else
    fprintf("Uyari: %s yok. Layer kaniti icin once trainLSTMModel calistir.\n", modelPath);
end

% -------------------------------------------------------------------------
% 2) DURATION DAGILIMI (duration_distribution.png)
% -------------------------------------------------------------------------
try
    load(seqPath, "allDurSeqs","allDurClassSeqs","durRep","numDurClasses");
    rawDur = cell2mat(allDurSeqs);
    durCls = cell2mat(allDurClassSeqs);

    fig = figure('Visible','off');
    subplot(2,1,1);
    histogram(rawDur, 60);
    xlabel("Duration (sec)"); ylabel("Count");
    title("Raw Duration Distribution");

    subplot(2,1,2);
    edges = 0.5:1:(numDurClasses+0.5);
    histogram(durCls, edges);
    xlim([1 numDurClasses]);
    xlabel("Duration Class"); ylabel("Count");
    title("Quantized Duration Class Distribution");

    hold on;
    yyaxis right;
    plot(1:numDurClasses, durRep, "-o");
    ylabel("Representative Duration (median)");
    yyaxis left;
    hold off;

    exportgraphics(fig, fullfile(outDir,"duration_distribution.png"));
    close(fig);
catch ME
    fprintf("Duration grafikleri üretilemedi: %s\n", ME.message);
end

% -------------------------------------------------------------------------
% 3) NOTA-ZAMAN (note_time_generated.png)
% -------------------------------------------------------------------------
try
    GG = load(genPath);

    if isfield(GG,"pitchSeq")
        pitchSeq = double(GG.pitchSeq);
    elseif isfield(GG,"generatedPitches")
        pitchSeq = double(GG.generatedPitches);
    else
        error("Nota-zaman icin pitch dizisi yok.");
    end

    if isfield(GG,"durSeq")
        durSeq = double(GG.durSeq);
    elseif isfield(GG,"generatedDur")
        durSeq = double(GG.generatedDur);
    else
        durSeq = 0.25 * ones(size(pitchSeq)); % fallback
    end

    t0 = [0, cumsum(durSeq(1:end-1))];
    t1 = t0 + durSeq;

    fig = figure('Visible','off'); hold on;
    for i = 1:numel(pitchSeq)
        plot([t0(i) t1(i)], [pitchSeq(i) pitchSeq(i)], "LineWidth", 2);
    end
    xlabel("Time (sec)"); ylabel("MIDI Pitch");
    title("Generated Melody: Note-Time Plot");
    grid on; hold off;

    exportgraphics(fig, fullfile(outDir,"note_time_generated.png"));
    close(fig);
catch ME
    fprintf("Nota-zaman grafiği üretilemedi: %s\n", ME.message);
end

% -------------------------------------------------------------------------
% 4) SPLIT SAYILARI + TEST PERFORMANSI (song-level 80/10/10)
% -------------------------------------------------------------------------
try
    load(seqPath, ...
        "allPitchSeqs","allDurClassSeqs","pitch2idx","pitchVocabSize","numDurClasses");

    tokenVocabSize = pitchVocabSize * numDurClasses;

    trainRatio = 0.80; valRatio = 0.10; %#ok<NASGU>
    nSongs = numel(allPitchSeqs);
    perm  = randperm(nSongs);

    nTrainSongs = floor(trainRatio*nSongs);
    nValSongs   = floor(valRatio*nSongs);

    trainSongs = perm(1:nTrainSongs);
    valSongs   = perm(nTrainSongs+1:nTrainSongs+nValSongs);
    testSongs  = perm(nTrainSongs+nValSongs+1:end);

    [XTrain, YTrain] = buildWindowsFromSongs(allPitchSeqs, allDurClassSeqs, pitch2idx, pitchVocabSize, ...
                                            trainSongs, songSplit_windowLen, tokenVocabSize, maxWindowsPerSet);
    [XVal,   YVal]   = buildWindowsFromSongs(allPitchSeqs, allDurClassSeqs, pitch2idx, pitchVocabSize, ...
                                            valSongs,   songSplit_windowLen, tokenVocabSize, maxWindowsPerSet);
    [XTest,  YTest]  = buildWindowsFromSongs(allPitchSeqs, allDurClassSeqs, pitch2idx, pitchVocabSize, ...
                                            testSongs,  songSplit_windowLen, tokenVocabSize, maxWindowsPerSet);

    % Split sayıları TXT
    txtPath = fullfile(outDir,"split_counts.txt");
    fid = fopen(txtPath,"w");
    fprintf(fid, "=== SPLIT COUNTS (SONG-LEVEL) ===\n");
    fprintf(fid, "Saved at: %s\n\n", datestr(now));
    fprintf(fid, "Songs: train=%d, val=%d, test=%d, total=%d\n", ...
        numel(trainSongs), numel(valSongs), numel(testSongs), nSongs);
    fprintf(fid, "windowLen=%d\n", songSplit_windowLen);
    fprintf(fid, "Windows (limited by maxWindowsPerSet=%d, 0=no limit):\n", maxWindowsPerSet);
    fprintf(fid, "  XTrain=%d\n  XVal=%d\n  XTest=%d\n", numel(XTrain), numel(XVal), numel(XTest));
    fclose(fid);

    % Split sayıları PNG
    fig = figure('Visible','off');
    bar([numel(trainSongs) numel(valSongs) numel(testSongs)]);
    set(gca,'XTickLabel',{'Train','Val','Test'});
    ylabel("Song Count");
    title("Song Split Counts (80/10/10)");
    grid on;
    exportgraphics(fig, fullfile(outDir,"split_counts.png"));
    close(fig);

    if doSongSplitTrainAndTest
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

        options = trainingOptions("adam", ...
            "InitialLearnRate", 1e-3, ...
            "MaxEpochs", songSplit_maxEpochs, ...
            "MiniBatchSize", songSplit_miniBatch, ...
            "Shuffle", "every-epoch", ...
            "ValidationData", {XVal, YVal}, ...
            "ValidationFrequency", 200, ...
            "Verbose", true);

        net2 = trainNetwork(XTrain, YTrain, layers, options);

        [acc, nll, ppl] = evalNextTokenMetrics(net2, XTest, YTest);

        outPath = fullfile(outDir, "test_metrics_songSplit.txt");
        fid = fopen(outPath, "w");
        fprintf(fid, "=== TEST METRICS (SONG-LEVEL 80/10/10) ===\n");
        fprintf(fid, "Saved at: %s\n\n", datestr(now));
        fprintf(fid, "Num test windows: %d\n", numel(XTest));
        fprintf(fid, "Accuracy (top-1): %.4f\n", acc);
        fprintf(fid, "Avg NLL (cross-entropy): %.6f\n", nll);
        fprintf(fid, "Perplexity: %.4f\n", ppl);
        fclose(fid);

        try
            save(fullfile("models","lstmModel_nottingham_songSplit.mat"), "net2", "tokenVocabSize");
        catch
        end
    end

catch ME
    fprintf("Song-split/test bloğu başarısız: %s\n", ME.message);
end

% -------------------------------------------------------------------------
% 5) WINDOW LENGTH ABLATION (window_length_ablation.png)
% -------------------------------------------------------------------------
if doWindowLengthAblation
    try
        load(seqPath, ...
            "allPitchSeqs","allDurClassSeqs","pitch2idx","pitchVocabSize","numDurClasses");
        tokenVocabSize = pitchVocabSize * numDurClasses;

        trainRatio = 0.80; valRatio = 0.10;
        nSongs = numel(allPitchSeqs);
        perm  = randperm(nSongs);
        nTrainSongs = floor(trainRatio*nSongs);
        nValSongs   = floor(valRatio*nSongs);
        trainSongs = perm(1:nTrainSongs);
        valSongs   = perm(nTrainSongs+1:nTrainSongs+nValSongs);

        valLosses = zeros(size(ablation_windowLens));
        valAccs   = zeros(size(ablation_windowLens));

        for w = 1:numel(ablation_windowLens)
            windowLen = ablation_windowLens(w);

            [XTrainA, YTrainA] = buildWindowsFromSongs(allPitchSeqs, allDurClassSeqs, pitch2idx, pitchVocabSize, ...
                                                      trainSongs, windowLen, tokenVocabSize, maxWindowsPerSet);
            [XValA,   YValA]   = buildWindowsFromSongs(allPitchSeqs, allDurClassSeqs, pitch2idx, pitchVocabSize, ...
                                                      valSongs,   windowLen, tokenVocabSize, maxWindowsPerSet);

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

            options = trainingOptions("adam", ...
                "InitialLearnRate", 1e-3, ...
                "MaxEpochs", ablation_epochs, ...
                "MiniBatchSize", ablation_miniBatch, ...
                "Shuffle", "every-epoch", ...
                "ValidationData", {XValA, YValA}, ...
                "ValidationFrequency", 200, ...
                "Verbose", false);

            netA = trainNetwork(XTrainA, YTrainA, layers, options);

            [accA, nllA] = evalNextTokenMetrics(netA, XValA, YValA);
            valAccs(w)   = accA;
            valLosses(w) = nllA;

            fprintf("Ablation windowLen=%d | valAcc=%.4f | valLoss=%.4f\n", windowLen, accA, nllA);
        end

        fig = figure('Visible','off');
        yyaxis left;
        plot(ablation_windowLens, valAccs, "-o", "LineWidth", 2);
        ylabel("Validation Accuracy");
        yyaxis right;
        plot(ablation_windowLens, valLosses, "-o", "LineWidth", 2);
        ylabel("Validation NLL (Loss)");
        xlabel("Window Length");
        title("Window Length Ablation (Song-Level Split)");
        grid on;

        exportgraphics(fig, fullfile(outDir,"window_length_ablation.png"));
        close(fig);

        save(fullfile(outDir,"window_length_ablation.mat"), ...
            "ablation_windowLens","valAccs","valLosses","ablation_epochs","ablation_miniBatch","maxWindowsPerSet");

    catch ME
        fprintf("Ablation başarısız: %s\n", ME.message);
    end
end

% -------------------------------------------------------------------------
% 6) (OPSİYONEL) HARİCİ MIDI TEST — data/external_test.mid varsa
% -------------------------------------------------------------------------
try
    extMidi = fullfile(fileparts(mfilename('fullpath')), "external","matlab-midi","tests","jesu.mid");
    if isfile(extMidi)
        load(seqPath, "pitch2idx","pitchVocabSize","durEdges","numDurClasses");
        modelPath2 = fullfile("models","lstmModel_nottingham.mat");
        if isfile(modelPath2)
            S = load(modelPath2,"net");
            net = S.net;

            [pSeq, dSeq] = midiToPitchSeq(extMidi);
            if isempty(pSeq) || isempty(dSeq)
                error("external_test.mid bos/okunamadi.");
            end

            dClass = discretize(dSeq, durEdges);

            pIdx = zeros(size(pSeq));
            ok = true(size(pSeq));
            for i = 1:numel(pSeq)
                if isKey(pitch2idx, pSeq(i))
                    pIdx(i) = pitch2idx(pSeq(i));
                else
                    ok(i) = false;
                end
            end

            pIdx = pIdx(ok);
            dClass = dClass(ok);

            tokens = pIdx + (double(dClass)-1)*pitchVocabSize;

            if numel(tokens) > songSplit_windowLen
                [Xext, YextNum] = buildWindowsFromTokens(tokens, songSplit_windowLen);
                Yext = categorical(YextNum(:), 1:(pitchVocabSize*numDurClasses));

                [accE, nllE, pplE] = evalNextTokenMetrics(net, Xext, Yext);

                outPath = fullfile(outDir,"external_midi_test_metrics.txt");
                fid = fopen(outPath,"w");
                fprintf(fid, "=== EXTERNAL MIDI TEST ===\n");
                fprintf(fid, "File: %s\n", extMidi);
                fprintf(fid, "Saved at: %s\n\n", datestr(now));
                fprintf(fid, "Num windows: %d\n", numel(Xext));
                fprintf(fid, "Accuracy (top-1): %.4f\n", accE);
                fprintf(fid, "Avg NLL: %.6f\n", nllE);
                fprintf(fid, "Perplexity: %.4f\n", pplE);
                fclose(fid);
            end
        end
    end
catch
end

fprintf("KANIT PAKETI bitti. results/ klasörüne bak.\n");


%% ======================= LOCAL FUNCTIONS ===============================
function [Xset, Yset] = buildWindowsFromSongs(allPitchSeqs, allDurClassSeqs, pitch2idx, pitchVocabSize, ...
                                             songIdxList, windowLen, tokenVocabSize, maxWindows)
    Xset = {};
    Yall = [];

    for si = 1:numel(songIdxList)
        s = songIdxList(si);

        p = allPitchSeqs{s};
        d = allDurClassSeqs{s};

        pIdx = zeros(size(p));
        for i = 1:numel(p)
            pIdx(i) = pitch2idx(p(i));
        end

        tokens = pIdx + (double(d)-1)*pitchVocabSize;

        if numel(tokens) <= windowLen
            continue;
        end

        [X, Y] = buildWindowsFromTokens(tokens, windowLen);

        Xset = [Xset, X]; %#ok<AGROW>
        Yall = [Yall, Y]; %#ok<AGROW>

        if maxWindows > 0 && numel(Xset) >= maxWindows
            break;
        end
    end

    if maxWindows > 0 && numel(Xset) > maxWindows
        Xset = Xset(1:maxWindows);
        Yall = Yall(1:maxWindows);
    end

    Yset = categorical(Yall(:), 1:tokenVocabSize);
end

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
