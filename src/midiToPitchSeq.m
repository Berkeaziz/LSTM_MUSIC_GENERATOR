
function [pitchSeq, durSeq] = midiToPitchSeq(midiFilePath)
% midiToPitchSeq  Monophonic MIDI dosyasından pitch ve duration dizileri çıkarır.
% pitchSeq: 1xN double (MIDI pitch)
% durSeq  : 1xN double (notanın süresi; midiInfo'nun zaman biriminde)


    midiStruct = readmidi(midiFilePath);
    notes = midiInfo(midiStruct, 0);

    if isempty(notes)
        pitchSeq = [];
        durSeq   = [];
        return;
    end

    % MIDI Toolbox tipik format: [track chan pitch vel start end]
    onset  = notes(:,5);
    lastCol = notes(:,6);

    % Bazı sürümlerde 6. sütun "end", bazılarında "duration" olabiliyor.
    if all(lastCol >= onset)
        dur = lastCol - onset;    % end-start
    else
        dur = lastCol;            % duration
    end

    % onset'e göre sırala
    [~, idx] = sort(onset, "ascend");
    notes = notes(idx,:);
    dur   = dur(idx);

    pitches = round(notes(:,3));

    pitchSeq = pitches(:).';
    durSeq   = dur(:).';

    % Sıfır / negatif süreleri ayıkla (nadiren bozuk satır çıkabiliyor)
    bad = durSeq <= 0;
    if any(bad)
        pitchSeq(bad) = [];
        durSeq(bad)   = [];
    end
end
