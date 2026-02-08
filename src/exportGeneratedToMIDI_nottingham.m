%% exportGeneratedToMIDI_nottingham.m
% Amaç: Üretilen melodiyi (pitch + opsiyonel duration) MIDI dosyasına yazmak.

clear; clc;

% --- Proje köküne geç (script src içindeyse) ---
thisFile = mfilename("fullpath");
projectRoot = fileparts(fileparts(thisFile)); % src -> proje kökü
cd(projectRoot);

% --- matlab-midi kütüphanesini path'e ekle ---
midiLib = fullfile(projectRoot,"external","matlab-midi","src");
assert(exist(midiLib,"dir")==7, "MIDI lib klasörü yok: %s", midiLib);
addpath(genpath(midiLib));

assert(~isempty(which("matrix2midi")), "matrix2midi bulunamadı. Path ekleme başarısız.");
assert(~isempty(which("writemidi")),  "writemidi bulunamadı. Path ekleme başarısız.");

% --- Üretilen melodiyi yükle ---
genPath = fullfile("results","generatedMelody_nottingham.mat");
assert(isfile(genPath), "Bulunamadı: %s (Önce generateMelody_nottingham.m çalıştır)", genPath);

S = load(genPath);

% generatedPitches kontrol
assert(isfield(S,"generatedPitches"), "MAT dosyasında generatedPitches yok.");
generatedPitches = double(S.generatedPitches(:)'); % satır vektör

% duration varsa kullan, yoksa sabit süre
if isfield(S,"generatedDur") && ~isempty(S.generatedDur)
    generatedDur = double(S.generatedDur(:)'); % saniye
    if numel(generatedDur) ~= numel(generatedPitches)
        warning("generatedDur uzunluğu pitch ile eşleşmiyor. Sabit süreye düşüyorum.");
        generatedDur = [];
    end
else
    generatedDur = [];
end

% --- MIDI ayarları ---
% Tempo tanımı: matlab-midi genelde "beats" tabanlı çalışır. Biz süreleri doğrudan saniye gibi
% kullanmak yerine, bir "beat" = 1 birim kabul edip tempo ile ölçeklemek daha sağlıklı.
% Basit yol: beatDur = 0.5s (120 BPM) gibi bir ölçek seç, süreleri beat cinsine çevir.

tempoBPM = 120;              % MIDI meta tempo
beatSec  = 60/tempoBPM;      % 1 beat kaç saniye
defaultNoteSec = 0.5;        % duration yoksa her notaya bu (s)

velocity = 90;               % 1..127
channel  = 1;                % 1..16

% --- Note matrix oluştur: [track channel pitch velocity start end] ---
N = numel(generatedPitches);
noteMat = zeros(N, 6);

tBeat = 0; % beat cinsinden zaman

for i = 1:N
    p = generatedPitches(i);

    % MIDI pitch aralığını clamp'le (0..127)
    p = max(0, min(127, round(p)));

    % süre (beat cinsinden)
    if isempty(generatedDur)
        durBeat = defaultNoteSec / beatSec;
    else
        durSec  = max(0.05, generatedDur(i));   % aşırı kısa/0 olmasın
        durBeat = durSec / beatSec;
    end

    startBeat = tBeat;
    endBeat   = tBeat + durBeat;

    noteMat(i,:) = [1, channel, p, velocity, startBeat, endBeat];

    tBeat = endBeat; % legato; istersen araya boşluk ekleyebilirsin
end

% --- MIDI yaz ---
midiStruct = matrix2midi(noteMat);
outMidPath = fullfile("results","generatedMelody_nottingham.mid");
writemidi(midiStruct, outMidPath);

fprintf("MIDI yazıldı: %s\n", outMidPath);
fprintf("Not: tempo = %d BPM, note count = %d\n", tempoBPM, N);
