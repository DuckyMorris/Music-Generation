from music21 import *
import numpy as np
path = "Piano G/"
import os

def convert_to_midi(prediction_output):
    '''
    Converts our predicted output to notes and then writes it to a midi file
    '''
    offset = 0
    out = []

    for progression in prediction_output:

        if ('.' in progression) or (progression.isdigit()):

            chord = progression.split('.')
            notes = []

            for curr in chord:
                cn=int(curr)
                new = note.Note(cn)
                new.storedInstrument = instrument.Piano()
                notes.append(new)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            out.append(new_chord)

        else:
            new = note.Note(progression)
            new.offset = offset
            new.storedInstrument = instrument.Piano()
            out.append(new)

        offset += 0.5

    s= stream.Stream(out)
    s.write('midi', fp='music.mid')



def read_midi(file):
    '''
    Function that takes an input string which is a string that has the location of a midi file and converts it to a seqeunce of notes and returns that
    '''

    print("Loading File:", file)

    n = []

    parsed_file = converter.parse(file)

    partitions = instrument.partitionByInstrument(parsed_file)

    for instrument_ in partitions.parts:
        #print(str(part))
        if 'Piano' in str(instrument_) or "PIANO" in str(instrument_) or "BASS" in str(instrument_) or "BASSOON" in str(instrument_):

            notes = instrument_.recurse()

            for note_ in notes:
                try:

                    if isinstance(note_, note.Note):
                        n.append(str(note_.pitch))
                except:
                    if isinstance(note_, chord.Chord):
                        n.append('.'.join(str(n) for n in note_.normalOrder))


    return np.array(n)
