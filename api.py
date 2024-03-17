import argparse
import os
import random
from urllib import request

import torch
import torch.nn.functional as F
import progressbar
import torchaudio

from models.classifier import AudioMiniEncoderWithClassifierHead
from models.cvvp import CVVP
from models.diffusion_decoder import DiffusionTts
from models.autoregressive import UnifiedVoice
from tqdm import tqdm

from models.arch_util import TorchMelSpectrogram
from models.clvp import CLVP
from models.vocoder import UnivNetGenerator
from utils.audio import load_audio, wav_to_univnet_mel, denormalize_tacotron_mel
from utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from utils.tokenizer import VoiceBpeTokenizer, lev_distance


pbar = None


def download_models(specific_models=None):
    """
    Call to download all the models that Tortoise uses.
    """
    MODELS = {
        'autoregressive.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/hf/.models/autoregressive.pth',
        'classifier.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/hf/.models/classifier.pth',
        'clvp.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/hf/.models/clvp.pth',
        'cvvp.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/hf/.models/cvvp.pth',
        'diffusion_decoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/hf/.models/diffusion_decoder.pth',
        'vocoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/hf/.models/vocoder.pth',
    }
    os.makedirs('.models', exist_ok=True)
    def show_progress(block_num, block_size, total_size):
        global pbar
        if pbar is None:
            pbar = progressbar.ProgressBar(maxval=total_size)
            pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None
    for model_name, url in MODELS.items():
        if specific_models is not None and model_name not in specific_models:
            continue
        if os.path.exists(f'.models/{model_name}'):
            continue
        print(f'Downloading {model_name} from {url}...')
        request.urlretrieve(url, f'.models/{model_name}', show_progress)
        print('Done.')


def pad_or_truncate(t, length):
    """
    Utility function for forcing <t> to have the specified sequence length, whether by clipping it or padding it with 0s.
    """
    if t.shape[-1] == length:
        return t
    elif t.shape[-1] < length:
        return F.pad(t, (0, length-t.shape[-1]))
    else:
        return t[..., :length]


def load_discrete_vocoder_diffuser(trained_diffusion_steps=4000, desired_diffusion_steps=200, cond_free=True, cond_free_k=1):
    """
    Helper function to load a GaussianDiffusion instance configured for use as a vocoder.
    """
    return SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=cond_free, conditioning_free_k=cond_free_k)


def format_conditioning(clip, cond_length=132300):
    """
    Converts the given conditioning signal to a MEL spectrogram and clips it as expected by the models.
    """
    gap = clip.shape[-1] - cond_length
    if gap < 0:
        clip = F.pad(clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        clip = clip[:, rand_start:rand_start + cond_length]
    mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).cuda()


def fix_autoregressive_output(codes, stop_token, complain=True):
    """
    This function performs some padding on coded audio that fixes a mismatch issue between what the diffusion model was
    trained on and what the autoregressive code generator creates (which has no padding or end).
    This is highly specific to the DVAE being used, so this particular coding will not necessarily work if used with
    a different DVAE. This can be inferred by feeding a audio clip padded with lots of zeros on the end through the DVAE
    and copying out the last few codes.

    Failing to do this padding will produce speech with a harsh end that sounds like "BLAH" or similar.
    """
    # Strip off the autoregressive stop token and add padding.
    stop_token_indices = (codes == stop_token).nonzero()
    if len(stop_token_indices) == 0:
        if complain:
            print("No stop tokens found, enjoy that output of yours!")
        return codes
    else:
        codes[stop_token_indices] = 83
    stm = stop_token_indices.min().item()
    codes[stm:] = 83
    if stm - 3 < codes.shape[0]:
        codes[-3] = 45
        codes[-2] = 45
        codes[-1] = 248

    return codes


def do_spectrogram_diffusion(diffusion_model, diffuser, latents, conditioning_samples, temperature=1, verbose=True):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        cond_mels = []
        for sample in conditioning_samples:
            # The diffuser operates at a sample rate of 24000 (except for the latent inputs)
            sample = torchaudio.functional.resample(sample, 22050, 24000)
            sample = pad_or_truncate(sample, 102400)
            cond_mel = wav_to_univnet_mel(sample.to(latents.device), do_normalization=False)
            cond_mels.append(cond_mel)
        cond_mels = torch.stack(cond_mels, dim=1)

        output_seq_len = latents.shape[1] * 4 * 24000 // 22050  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_shape = (latents.shape[0], 100, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(latents, cond_mels, output_seq_len, False)

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                      model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
                                     progress=verbose)
        return denormalize_tacotron_mel(mel)[:,:,:output_seq_len]


def classify_audio_clip(clip):
    """
    Returns whether or not Tortoises' classifier thinks the given clip came from Tortoise.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: True if the clip was classified as coming from Tortoise and false if it was classified as real.
    """
    download_models(['classifier.pth'])
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    classifier.load_state_dict(torch.load('models/classifier.pth', map_location=torch.device('cpu')))
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]


class TextToSpeech:
    """
    Main entry point into Tortoise.
    :param autoregressive_batch_size: Specifies how many samples to generate per batch. Lower this if you are seeing
                                      GPU OOM errors. Larger numbers generates slightly faster.
    """
    def __init__(self, autoregressive_batch_size=16):
        self.autoregressive_batch_size = autoregressive_batch_size
        self.tokenizer = VoiceBpeTokenizer()
        download_models()

        self.autoregressive = UnifiedVoice(max_mel_tokens=604, max_text_tokens=402, max_conditioning_inputs=2, layers=30,
                                      model_dim=1024,
                                      heads=16, number_text_tokens=255, start_text_token=255, checkpointing=False,
                                      train_solo_embeddings=False,
                                      average_conditioning_embeddings=True).cpu().eval()
        self.autoregressive.load_state_dict(torch.load('.models/autoregressive.pth'))

        self.clvp = CLVP(dim_text=512, dim_speech=512, dim_latent=512, num_text_tokens=256, text_enc_depth=12,
                         text_seq_len=350, text_heads=8,
                         num_speech_tokens=8192, speech_enc_depth=12, speech_heads=8, speech_seq_len=430,
                         use_xformers=True).cpu().eval()
        self.clvp.load_state_dict(torch.load('.models/clvp.pth'))

        self.cvvp = CVVP(model_dim=512, transformer_heads=8, dropout=0, mel_codes=8192, conditioning_enc_depth=8, cond_mask_percentage=0,
                         speech_enc_depth=8, speech_mask_percentage=0, latent_multiplier=1).cpu().eval()
        self.cvvp.load_state_dict(torch.load('.models/cvvp.pth'))

        self.diffusion = DiffusionTts(model_channels=1024, num_layers=10, in_channels=100, out_channels=200,
                                      in_latent_channels=1024, in_tokens=8193, dropout=0, use_fp16=False, num_heads=16,
                                      layer_drop=0, unconditioned_percentage=0).cpu().eval()
        self.diffusion.load_state_dict(torch.load('.models/diffusion_decoder.pth'))

        self.vocoder = UnivNetGenerator().cpu()
        self.vocoder.load_state_dict(torch.load('.models/vocoder.pth')['model_g'])
        self.vocoder.eval(inference=True)

    def tts_with_preset(self, text, voice_samples, preset='fast', **kwargs):
        """
        Calls TTS with one of a set of preset generation parameters. Options:
            'ultra_fast': Produces speech at a speed which belies the name of this repo. (Not really, but it's definitely fastest).
            'fast': Decent quality speech at a decent inference rate. A good choice for mass inference.
            'standard': Very good quality. This is generally about as good as you are going to get.
            'high_quality': Use if you want the absolute best. This is not really worth the compute, though.
        """
        # Use generally found best tuning knobs for generation.
        kwargs.update({'temperature': .8, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
                       #'typical_sampling': True,
                       'top_p': .8,
                       'cond_free_k': 2.0, 'diffusion_temperature': 1.0})
        # Presets are defined here.
        presets = {
            'ultra_fast': {'num_autoregressive_samples': 32, 'diffusion_iterations': 16, 'cond_free': False},
            'fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 32},
            'standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 128},
            'high_quality': {'num_autoregressive_samples': 512, 'diffusion_iterations': 1024},
        }
        kwargs.update(presets[preset])
        return self.tts(text, voice_samples, **kwargs)

    def tts(self, text, voice_samples, k=1, verbose=True,
            # autoregressive generation parameters follow
            num_autoregressive_samples=512, temperature=.8, length_penalty=1, repetition_penalty=2.0, top_p=.8, max_mel_tokens=500,
            typical_sampling=False, typical_mass=.9,
            # CLVP & CVVP parameters
            clvp_cvvp_slider=.5,
            # diffusion generation parameters follow
            diffusion_iterations=100, cond_free=True, cond_free_k=2, diffusion_temperature=1.0,
            **hf_generate_kwargs):
        """
        Produces an audio clip of the given text being spoken with the given reference voice.
        :param text: Text to be spoken.
        :param voice_samples: List of 2 or more ~10 second reference clips which should be torch tensors containing 22.05kHz waveform data.
        :param k: The number of returned clips. The most likely (as determined by Tortoises' CLVP and CVVP models) clips are returned.
        :param verbose: Whether or not to print log messages indicating the progress of creating a clip. Default=true.
        ~~AUTOREGRESSIVE KNOBS~~
        :param num_autoregressive_samples: Number of samples taken from the autoregressive model, all of which are filtered using CLVP+CVVP.
               As Tortoise is a probabilistic model, more samples means a higher probability of creating something "great".
        :param temperature: The softmax temperature of the autoregressive model.
        :param length_penalty: A length penalty applied to the autoregressive decoder. Higher settings causes the model to produce more terse outputs.
        :param repetition_penalty: A penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence
                                   of long silences or "uhhhhhhs", etc.
        :param top_p: P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely" (aka boring) outputs.
        :param max_mel_tokens: Restricts the output length. (0,600] integer. Each unit is 1/20 of a second.
        :param typical_sampling: Turns typical sampling on or off. This sampling mode is discussed in this paper: https://arxiv.org/abs/2202.00666
                                 I was interested in the premise, but the results were not as good as I was hoping. This is off by default, but
                                 could use some tuning.
        :param typical_mass: The typical_mass parameter from the typical_sampling algorithm.
        ~~CLVP-CVVP KNOBS~~
        :param clvp_cvvp_slider: Controls the influence of the CLVP and CVVP models in selecting the best output from the autoregressive model.
                                [0,1]. Values closer to 1 will cause Tortoise to emit clips that follow the text more. Values closer to
                                0 will cause Tortoise to emit clips that more closely follow the reference clip (e.g. the voice sounds more
                                similar).
        ~~DIFFUSION KNOBS~~
        :param diffusion_iterations: Number of diffusion steps to perform. [0,4000]. More steps means the network has more chances to iteratively refine
                                     the output, which should theoretically mean a higher quality output. Generally a value above 250 is not noticeably better,
                                     however.
        :param cond_free: Whether or not to perform conditioning-free diffusion. Conditioning-free diffusion performs two forward passes for
                          each diffusion step: one with the outputs of the autoregressive model and one with no conditioning priors. The output
                          of the two is blended according to the cond_free_k value below. Conditioning-free diffusion is the real deal, and
                          dramatically improves realism.
        :param cond_free_k: Knob that determines how to balance the conditioning free signal with the conditioning-present signal. [0,inf].
                            As cond_free_k increases, the output becomes dominated by the conditioning-free signal.
                            Formula is: output=cond_present_output*(cond_free_k+1)-cond_absenct_output*cond_free_k
        :param diffusion_temperature: Controls the variance of the noise fed into the diffusion model. [0,1]. Values at 0
                                      are the "mean" prediction of the diffusion network and will sound bland and smeared.
        ~~OTHER STUFF~~
        :param hf_generate_kwargs: The huggingface Transformers generate API is used for the autoregressive transformer.
                                   Extra keyword args fed to this function get forwarded directly to that API. Documentation
                                   here: https://huggingface.co/docs/transformers/internal/generation_utils
        :return: Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
                 Sample rate is 24kHz.
        """
        text = torch.IntTensor(self.tokenizer.encode(text)).unsqueeze(0).cuda()
        text = F.pad(text, (0, 1))  # This may not be necessary.

        conds = []
        if not isinstance(voice_samples, list):
            voice_samples = [voice_samples]
        for vs in voice_samples:
            conds.append(format_conditioning(vs))
        conds = torch.stack(conds, dim=1)

        diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=diffusion_iterations, cond_free=cond_free, cond_free_k=cond_free_k)

        with torch.no_grad():
            samples = []
            num_batches = num_autoregressive_samples // self.autoregressive_batch_size
            stop_mel_token = self.autoregressive.stop_mel_token
            calm_token = 83  # This is the token for coding silence, which is fixed in place with "fix_autoregressive_output"
            self.autoregressive = self.autoregressive.cuda()
            if verbose:
                print("Generating autoregressive samples..")
            for b in tqdm(range(num_batches), disable=not verbose):
                codes = self.autoregressive.inference_speech(conds, text,
                                                             do_sample=True,
                                                             top_p=top_p,
                                                             temperature=temperature,
                                                             num_return_sequences=self.autoregressive_batch_size,
                                                             length_penalty=length_penalty,
                                                             repetition_penalty=repetition_penalty,
                                                             max_generate_length=max_mel_tokens,
                                                             **hf_generate_kwargs)
                padding_needed = max_mel_tokens - codes.shape[1]
                codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
                samples.append(codes)
            self.autoregressive = self.autoregressive.cpu()

            clip_results = []
            self.clvp = self.clvp.cuda()
            self.cvvp = self.cvvp.cuda()
            if verbose:
                print("Computing best candidates using CLVP and CVVP")
            for batch in tqdm(samples, disable=not verbose):
                for i in range(batch.shape[0]):
                    batch[i] = fix_autoregressive_output(batch[i], stop_mel_token)
                clvp = self.clvp(text.repeat(batch.shape[0], 1), batch, return_loss=False)
                cvvp_accumulator = 0
                for cl in range(conds.shape[1]):
                    cvvp_accumulator = cvvp_accumulator + self.cvvp(conds[:, cl].repeat(batch.shape[0], 1, 1), batch, return_loss=False )
                cvvp = cvvp_accumulator / conds.shape[1]
                clip_results.append(clvp * clvp_cvvp_slider + cvvp * (1-clvp_cvvp_slider))
            clip_results = torch.cat(clip_results, dim=0)
            samples = torch.cat(samples, dim=0)
            best_results = samples[torch.topk(clip_results, k=k).indices]
            self.clvp = self.clvp.cpu()
            self.cvvp = self.cvvp.cpu()
            del samples

            # The diffusion model actually wants the last hidden layer from the autoregressive model as conditioning
            # inputs. Re-produce those for the top results. This could be made more efficient by storing all of these
            # results, but will increase memory usage.
            self.autoregressive = self.autoregressive.cuda()
            best_latents = self.autoregressive(conds, text, torch.tensor([text.shape[-1]], device=conds.device), best_results,
                                               torch.tensor([best_results.shape[-1]*self.autoregressive.mel_length_compression], device=conds.device),
                                               return_latent=True, clip_inputs=False)
            self.autoregressive = self.autoregressive.cpu()

            if verbose:
                print("Transforming autoregressive outputs into audio..")
            wav_candidates = []
            self.diffusion = self.diffusion.cuda()
            self.vocoder = self.vocoder.cuda()
            for b in range(best_results.shape[0]):
                codes = best_results[b].unsqueeze(0)
                latents = best_latents[b].unsqueeze(0)

                # Find the first occurrence of the "calm" token and trim the codes to that.
                ctokens = 0
                for k in range(codes.shape[-1]):
                    if codes[0, k] == calm_token:
                        ctokens += 1
                    else:
                        ctokens = 0
                    if ctokens > 8:  # 8 tokens gives the diffusion model some "breathing room" to terminate speech.
                        latents = latents[:, :k]
                        break

                mel = do_spectrogram_diffusion(self.diffusion, diffuser, latents, voice_samples, temperature=diffusion_temperature, verbose=verbose)
                wav = self.vocoder.inference(mel)
                wav_candidates.append(wav.cpu())
            self.diffusion = self.diffusion.cpu()
            self.vocoder = self.vocoder.cpu()

            if len(wav_candidates) > 1:
                return wav_candidates
            return wav_candidates[0]
