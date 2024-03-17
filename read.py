import argparse
import os

import torch
import torch.nn.functional as F
import torchaudio

from api import TextToSpeech, format_conditioning
from utils.audio import load_audio, get_voices
from utils.tokenizer import VoiceBpeTokenizer


def split_and_recombine_text(texts, desired_length=200, max_len=300):
    # TODO: also split across '!' and '?'. Attempt to keep quotations together.
    texts = [s.strip() + "." for s in texts.split('.')]

    i = 0
    while i < len(texts):
        ltxt = texts[i]
        if len(ltxt) >= desired_length or i == len(texts)-1:
            i += 1
            continue
        if len(ltxt) + len(texts[i+1]) > max_len:
            i += 1
            continue
        texts[i] = f'{ltxt} {texts[i+1]}'
        texts.pop(i+1)
    return texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--textfile', type=str, help='A file containing the text to read.', default="data/riding_hood.txt")
    parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='pat')
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/longform/')
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='standard')
    parser.add_argument('--regenerate', type=str, help='Comma-separated list of clip numbers to re-generate, or nothing.', default=None)
    parser.add_argument('--voice_diversity_intelligibility_slider', type=float,
                        help='How to balance vocal diversity with the quality/intelligibility of the spoken text. 0 means highly diverse voice (not recommended), 1 means maximize intellibility',
                        default=.5)
    args = parser.parse_args()

    outpath = args.output_path
    voices = get_voices()
    selected_voices = args.voice.split(',')
    regenerate = args.regenerate
    if regenerate is not None:
        regenerate = [int(e) for e in regenerate.split(',')]
    for selected_voice in selected_voices:
        voice_outpath = os.path.join(outpath, selected_voice)
        os.makedirs(voice_outpath, exist_ok=True)

        with open(args.textfile, 'r', encoding='utf-8') as f:
            text = ''.join([l for l in f.readlines()])
        texts = split_and_recombine_text(text)
        tts = TextToSpeech()

        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]
        cond_paths = []
        for vsel in voice_sel:
            if vsel not in voices.keys():
                print(f'Error: voice {vsel} not available. Skipping.')
                continue
            cond_paths.extend(voices[vsel])
        if not cond_paths:
            print('Error: no valid voices specified. Try again.')

        conds = []
        for cond_path in cond_paths:
            c = load_audio(cond_path, 22050)
            conds.append(c)
        all_parts = []
        for j, text in enumerate(texts):
            if regenerate is not None and j not in regenerate:
                all_parts.append(load_audio(os.path.join(voice_outpath, f'{j}.wav'), 24000))
                continue
            gen = tts.tts_with_preset(text, conds, preset=args.preset, clvp_cvvp_slider=args.voice_diversity_intelligibility_slider)
            gen = gen.squeeze(0).cpu()
            torchaudio.save(os.path.join(voice_outpath, f'{j}.wav'), gen, 24000)
            all_parts.append(gen)
        full_audio = torch.cat(all_parts, dim=-1)
        torchaudio.save(os.path.join(voice_outpath, 'combined.wav'), full_audio, 24000)

