% pitchToWav_nottingham.m  (Audio Toolbox - note onset click FIX)
% - Her notayı önce üretir, sin^2 fade-in/fade-out uygular (klik/cızırtı azaltır)
% - audioDeviceWriter'a sabit frameSize blok gönderir
% - Notalar arası kısa gap ekler
% - İsteğe bağlı normalize eder

clear; clc;

inMat  = fullfile("results","generatedMelody_nottingham.mat");
outWav = fullfile("results","generatedMelody_nottingham_audioTB.wav");

S = load(inMat);
generatedPitches = S.generatedPitches(:)';

if isfield(S,"generatedDur")
    generatedDur = S.generatedDur(:)';    % beats gibi varsayacağız
    hasRhythm = true;
else
    hasRhythm = false;
end

Fs       = 44100;
tempoBPM = 120;
beatDur  = 60/tempoBPM;

% --- Ayarlar ---
frameSize    = 512;     % sabit blok
amp          = 0.10;    % düşük tut (clipping/cızırtı azalır)
gapSec       = 0.010;   % 10 ms boşluk
attackSec    = 0.040;   % 40 ms fade-in
releaseSec   = 0.080;   % 80 ms fade-out
normalizeWav = true;

gapSamp     = round(gapSec*Fs);
attackSamp  = round(attackSec*Fs);
releaseSamp = round(releaseSec*Fs);

% --- Audio Toolbox var mı? ---
hasAT = (exist("audioDeviceWriter","class") == 8) && (exist("audioOscillator","class") == 8);
if ~hasAT
    error("Audio Toolbox nesneleri bulunamadı (audioDeviceWriter/audioOscillator).");
end

deviceWriter = audioDeviceWriter("SampleRate", Fs);

osc = audioOscillator("sine", ...
    "SampleRate", Fs, ...
    "SamplesPerFrame", frameSize, ...
    "Amplitude", amp);

y = [];

for k = 1:numel(generatedPitches)
    p = generatedPitches(k);
    f = 440 * 2^((p-69)/12);

    if hasRhythm
        noteDurSec = max(0.06, generatedDur(k) * beatDur);
    else
        noteDurSec = max(0.10, 1.0 * beatDur);
    end

    N = max(1, round(noteDurSec * Fs));

    % Fade süreleri notadan uzun olmasın
    a = min(attackSamp,  max(1, floor(N/3)));
    r = min(releaseSamp, max(1, floor(N/3)));

    % Notayı üret (tam N örnek)
    reset(osc);              % <- fazı sıfırla (başlangıç kliklerini azaltır)
    osc.Frequency = f;

    noteAudio = zeros(1, N);
    pos = 1;
    nLeft = N;

    while nLeft > 0
        frame = osc();            % frameSize x 1
        frame = frame(:).';       % 1 x frameSize

        nThis = min(frameSize, nLeft);
        noteAudio(pos:pos+nThis-1) = frame(1:nThis);

        pos = pos + nThis;
        nLeft = nLeft - nThis;
    end

    % sin^2 (raised-cosine) fade-in / fade-out
    if a > 1
        fadeIn = sin(linspace(0, pi/2, a)).^2;
        noteAudio(1:a) = noteAudio(1:a) .* fadeIn;
        noteAudio(1) = 0; % ilk örnek kesin 0
    end
    if r > 1
        fadeOut = sin(linspace(pi/2, 0, r)).^2;
        noteAudio(end-r+1:end) = noteAudio(end-r+1:end) .* fadeOut;
        noteAudio(end) = 0; % son örnek kesin 0
    end

    % Real-time çal (sabit bloklarla) + y'ye ekle
    pos = 1;
    nLeft = N;
    while nLeft > 0
        nThis = min(frameSize, nLeft);

        frameOut = zeros(frameSize, 1);
        frameOut(1:nThis) = noteAudio(pos:pos+nThis-1).';

        deviceWriter(frameOut);

        pos = pos + nThis;
        nLeft = nLeft - nThis;
    end

    % notayı ve araya gap ekle
    y = [y, noteAudio, zeros(1, gapSamp)]; %#ok<AGROW>
end

release(deviceWriter);

% Normalize (WAV'da clipping olmasın)
if normalizeWav
    m = max(abs(y));
    if m > 0
        y = 0.95 * (y / m);
    end
end

if ~isfolder("results"), mkdir("results"); end
audiowrite(outWav, y(:), Fs);

fprintf("Bitti. WAV: %s\n", outWav);
