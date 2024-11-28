from datetime import timedelta
import logging
import time
import torch
import random
import librosa
import zipfile
import torchaudio
import numpy as np
import os,sys
import ffmpeg
import audiosegment
from srt import parse as SrtParse
from tqdm import tqdm
from cosyvoice.cli.cosyvoice import CosyVoice

from time import time as ttime
from modelscope import snapshot_download

from cvnodes.subtitle import Subtitle, load_srt, ms_to_srt_time, parse_srt_to_objects, read_file, save_subtitles_to_srt
from cvnodes.util import INPUT_DIR, OUTPUT_DIR, ROOT_DIR

sys.path.append(os.path.join(ROOT_DIR,'third_part'))
pretrained_models = os.path.join(ROOT_DIR,"pretrained_models")

input_dir = INPUT_DIR
output_dir = OUTPUT_DIR


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

sft_spk_list = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
tts_text_type = ['TEXT','SRT',]
voice_dir = os.path.join(INPUT_DIR,"voices")
logger.debug(f'voice dir:{voice_dir}')
def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

max_val = 0.8
prompt_sr, target_sr = 16000, 22050

def postprocess(speech, top_db=60, hop_length=220, win_length=440,target_sr=22050,max_val=1):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )

    if speech.size == 0:
        raise ValueError("All samples were trimmed as silence. Please check input or adjust 'top_db'.")

    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val

    if speech.dim() == 1:
        speech = speech.unsqueeze(0)  # 将 (N,) 转为 (1, N)

    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    
    return speech

def speed_change(input_audio, speed, sr):
    """
    Adjusts the speed of the input audio using FFmpeg.
    
    Args:
        input_audio (np.ndarray): Audio data in np.int16 format.
        speed (float): Speed factor (e.g., 1.2 for 20% faster).
        sr (int): Sampling rate of the input audio.

    Returns:
        np.ndarray: Processed audio with adjusted speed.
    """
    import ffmpeg

    raw_audio = None
    if input_audio.dtype != np.int16:
        logger.warning("Input audio data must be of type np.int16, default convert")
        # Convert the audio to raw byte format
        raw_audio = input_audio.astype(np.int16).tobytes()
    else:
        raw_audio = input_audio.tobytes()
    
    # Set up the FFmpeg input and apply atempo filter for speed change
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)
    output_stream = input_stream.filter('atempo', speed)
    
    # Capture the processed audio from FFmpeg
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(out, np.int16)


def load_cosyvoice(inference_mode,prompt_wav=None,prompt_text=None, instruct_text=None) ->CosyVoice:
    if inference_mode == '自然语言控制':
        model_dir = os.path.join(pretrained_models,"CosyVoice-300M-Instruct")
        snapshot_download(model_id="iic/CosyVoice-300M-Instruct",local_dir=model_dir)
        assert instruct_text is not None, "in 自然语言控制 mode, instruct_text can't be none"
    if inference_mode in ["跨语种复刻",'3s极速复刻']:
        model_dir = os.path.join(pretrained_models,"CosyVoice-300M")
        snapshot_download(model_id="iic/CosyVoice-300M",local_dir=model_dir)
        assert prompt_wav is not None, "in 跨语种复刻 or 3s极速复刻 mode, prompt_wav can't be none"
        if inference_mode == "3s极速复刻":
            assert len(prompt_text) > 0, "prompt文本为空，您是否忘记输入prompt文本？"
    if inference_mode == "预训练音色":
        model_dir = os.path.join(pretrained_models,"CosyVoice-300M-SFT")
        snapshot_download(model_id="iic/CosyVoice-300M-SFT",local_dir=model_dir)


    if model_dir != model_dir:
        model_dir = model_dir

    return CosyVoice(model_dir)

class CosyVoiceNode:
    def __init__(self):
        self.model_dir = None
        self.cosyvoice = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("TEXT",),
                "tts_text_type":(tts_text_type,{
                    "default":'TEXT'
                }),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "inference_mode":(inference_mode_list,{
                    "default": "预训练音色"
                }),
                "sft_dropdown":(sft_spk_list,{
                    "default":"中文女"
                }),
                "seed":("INT",{
                    "default": 42
                }),
            },
            "optional":{
                "speaker":(['无']+[f for f in os.listdir(voice_dir) if os.path.isfile(os.path.join(voice_dir, f)) and f.split('.')[-1] in ["pt"]],{
                    "default":'无'
                }),
                "tts_srt": ("SRT",),
                "prompt_text":("TEXT",),
                "prompt_wav": ("AUDIO",),
                "instruct_text":("TEXT",),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "ComfyUI-CosyVoice"

    def generate(self,tts_text,tts_text_type,speed,inference_mode,sft_dropdown,seed,speaker=None,tts_srt = None,
                 prompt_text=None,prompt_wav=None,instruct_text=None):
        t0 = ttime()
        if inference_mode == '自然语言控制':
            model_dir = os.path.join(pretrained_models,"CosyVoice-300M-Instruct")
            snapshot_download(model_id="iic/CosyVoice-300M-Instruct",local_dir=model_dir)
            assert instruct_text is not None, "in 自然语言控制 mode, instruct_text can't be none"
        if inference_mode in ["跨语种复刻",'3s极速复刻']:
            model_dir = os.path.join(pretrained_models,"CosyVoice-300M")
            snapshot_download(model_id="iic/CosyVoice-300M",local_dir=model_dir)
            assert prompt_wav is not None, "in 跨语种复刻 or 3s极速复刻 mode, prompt_wav can't be none"
            if inference_mode == "3s极速复刻":
                assert len(prompt_text) > 0, "prompt文本为空，您是否忘记输入prompt文本？"
        if inference_mode == "预训练音色":
            model_dir = os.path.join(pretrained_models,"CosyVoice-300M-SFT")
            snapshot_download(model_id="iic/CosyVoice-300M-SFT",local_dir=model_dir)


        if self.model_dir != model_dir:
            self.model_dir = model_dir
        
        logger.info(f'init model dir: {self.model_dir}')

        if self.cosyvoice is None:
            self.cosyvoice = load_cosyvoice(inference_mode,prompt_text=prompt_text,prompt_wav=prompt_wav,instruct_text=instruct_text)

        if prompt_wav:
            waveform = prompt_wav['waveform'].squeeze(0)
            source_sr = prompt_wav['sample_rate']

            num_samples = waveform.shape[0]  
            duration = num_samples / source_sr 
            logger.debug(f"sample number: {num_samples}, duration: {duration:.2f} s")

            # fix https://az-web.site/posts/2024/11/e83df2df.html#waveform-mean-%E9%9F%B3%E9%A2%91%E6%97%B6%E9%95%BF%E5%8F%98%E4%B8%BA-0-%E9%97%AE%E9%A2%98%E5%88%86%E6%9E%90%EF%BC%9A
            if waveform.dim() == 1:
                speech = waveform.unsqueeze(0)  # 保持 (1, N)
            else:  # 
                speech = waveform.mean(dim=0, keepdim=True)

            if source_sr != prompt_sr:
                speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
            
            logger.debug(f'prompt wave shape :{speech.shape} , rate:{prompt_sr}')

        if inference_mode == '预训练音色':
            logger.debug('get sft inference request')
            set_all_random_seed(seed)
            output = self.cosyvoice.inference_sft(tts_text, sft_dropdown)
        elif inference_mode == '3s极速复刻':
            logger.debug('get zero_shot inference request')
            prompt_speech_16k = postprocess(speech= speech,target_sr=target_sr,max_val=max_val)

            set_all_random_seed(seed)
            output = self.cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
        elif inference_mode == '跨语种复刻':
            logger.debug('get cross_lingual inference request')
            prompt_speech_16k = postprocess(speech= speech,target_sr=target_sr,max_val=max_val)
            set_all_random_seed(seed)
            output = self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
        else:
            logger.debug('get instruct inference request')
            set_all_random_seed(seed)
            logger.debug(self.model_dir)
            output = self.cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text)
        output_list = []
        for out_dict in output:
            output_numpy = out_dict['tts_speech'].squeeze(0).numpy() * 32768 
            output_numpy = output_numpy.astype(np.int16)
            if speed > 1.0 or speed < 1.0:
                output_numpy = speed_change(output_numpy,speed,target_sr)
            output_list.append(torch.Tensor(output_numpy/32768).unsqueeze(0))
        t1 = ttime()
        logger.debug("cost time \t %.3f" % (t1-t0))
        audio = {"waveform": torch.cat(output_list,dim=1).unsqueeze(0),"sample_rate":target_sr}
        return (audio,)


class CosyVoiceSrtNode:
    def __init__(self):
        self.model_dir = None
        self.cosyvoice = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "inference_mode":(inference_mode_list,{
                    "default": "预训练音色"
                }),
                "spk_id":(sft_spk_list,{
                    "default":"中文女"
                }),
            },
            "optional":{
                "tts_srt": ("SRT",),
                "srt_text":("TEXT",{"default":''}),
                "speaker":(['无']+[os.path.splitext(f)[0] for f in os.listdir(voice_dir) if os.path.isfile(os.path.join(voice_dir, f)) and f.endswith('.pt')],{
                    "default":'无'
                }),
            }
        }
    RETURN_TYPES = ("AUDIO","SRT")
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "ComfyUI-CosyVoice"

    def generate(self,speed,inference_mode,spk_id,speaker,tts_srt=None,srt_text=''):
        t0 = ttime()        
        assert inference_mode == "预训练音色", "请选择：预训练音色,预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！"

        if inference_mode == "预训练音色":
            model_dir = os.path.join(pretrained_models,"CosyVoice-300M-SFT")
            snapshot_download(model_id="iic/CosyVoice-300M-SFT",local_dir=model_dir)
            
        assert speaker , "音色为空请选择音色"

        if self.model_dir != model_dir:
            self.model_dir = model_dir
        
        logger.info(f'init model dir: {self.model_dir}')

        if self.cosyvoice is None:
            self.cosyvoice = load_cosyvoice(inference_mode)
        
        if tts_srt:
            tts_text = read_file(tts_srt)
        else:
            tts_text = srt_text
            print('tts srt is empty use srt_text')

        if inference_mode == '预训练音色':
            logger.debug('get sft inference request')
            
            output_list,new_subtitles = self.process_srt_inference(tts_text, spk_id=spk_id,stream=False,speed=speed,speaker=speaker)
            
            output_dir = OUTPUT_DIR
            # Save the new subtitle file
            srt_file = os.path.join(output_dir, "output_segments.srt")        
            save_subtitles_to_srt(new_subtitles,srt_file)
            logger.info(f"Saved subtitle file: {srt_file}")
            
            # Combine all audio clips into one
            if output_list:
                try:
                    combined_audio = torch.cat(output_list, dim=-1)
                    combined_audio_file = os.path.join(output_dir, "output_combined.wav")
                    torchaudio.save(combined_audio_file, combined_audio, 22050)
                    logger.info(f"Saved combined audio file: {combined_audio_file}")
                except Exception as e:
                    logger.error(f"Error combining or saving audio file: {e}")

            # output_list = []
            # for out_dict in output:
            #     output_numpy = out_dict['tts_speech'].squeeze(0).numpy() * 32768 
            #     output_numpy = output_numpy.astype(np.int16)
            #     if speed > 1.0 or speed < 1.0:
            #         output_numpy = speed_change(output_numpy,speed,target_sr)
            #     output_list.append(torch.Tensor(output_numpy/32768).unsqueeze(0))
            t1 = ttime()
            logger.debug("cost time \t %.3f" % (t1-t0))
            audio = {"waveform": combined_audio.unsqueeze(0),"sample_rate":target_sr}
            return (audio,srt_file)


    def process_srt_inference(self, srt_content, spk_id, stream=False, speed=1.0, speaker=None, output_dir=output_dir):
        """
        根据输入的 SRT 字幕内容生成对应的音频和新字幕文件。
        """
        inference_mode = '预训练音色'
        if self.cosyvoice is None:
            self.cosyvoice = load_cosyvoice(inference_mode)

        # Parse the SRT content into Subtitle objects
        subtitles = parse_srt_to_objects(srt_content)

        # Prepare for audio and subtitle saving
        combined_audio_segments = []
        new_subtitles = []
        audio_samples = 0  # 累积样本数
        subtitle_index = 1  # 字幕编号
        speaker_data = None

        if speaker not in (None, '无'):
            speaker_data = self.load_speaker(speaker)
        logger.info(f'spk_id:{spk_id},speaker:{speaker} len:{len(subtitles)}')
        os.makedirs(output_dir, exist_ok=True)
        
        for subtitle in (subtitles if subtitles else [Subtitle(1, "00:00:00,000", "00:00:00,000", srt_content)]):

            # Call the inference_sft method for each subtitle text      
            logger.info(f'process:{subtitle.text}')      
            output = self.cosyvoice.inference_sft(subtitle.text, spk_id, stream, speed, speaker_data,update_keys=None)

            for model_output in output:
                # Extract audio data
                audio_chunk = model_output["tts_speech"]  # 音频段数据
                text_chunk = model_output["text_chunk"]  # 对应的字幕
                

                start_time = ms_to_srt_time(audio_samples * 1000.0 / 22050)
                audio_samples += audio_chunk.shape[1]
                end_time = ms_to_srt_time(audio_samples * 1000.0 / 22050)
                new_subtitle = Subtitle.from_str_time(subtitle_index,start_time,end_time,text_chunk)
                new_subtitles.append(new_subtitle)
                logger.info(f'gererate text:{text_chunk}')

                audio_file = os.path.join(output_dir, f"{subtitle_index:03d}.wav")
                # Ensure audio_chunk is in the correct shape
                if audio_chunk.ndim == 1:  # If 1D (samples), add channel dimension
                    audio_chunk = audio_chunk.unsqueeze(0)

                torchaudio.save(audio_file, audio_chunk, 22050)
                logger.info(f"Saved audio file: {audio_file}")
                
                # Append the audio segment to the combined list
                combined_audio_segments.append(audio_chunk)
                
                subtitle_index += 1  

        logger.info(f"处理完成，共生成 {subtitle_index - 1} 段音频和字幕。")
        # Combine all audio clips into one
        if combined_audio_segments:
            try:
                combined_audio = torch.cat(combined_audio_segments, dim=-1)
                combined_audio_file = os.path.join(output_dir, "output_combined.wav")
                torchaudio.save(combined_audio_file, combined_audio, 22050)
                logger.info(f"Saved combined audio file: {combined_audio_file}")
            except Exception as e:
                logger.error(f"Error combining or saving audio file: {e}")

        return combined_audio_segments,new_subtitles

    def process_output(output):
        
        combined_audio_segments = []
        new_subtitles = []
        for model_output in output:
                # Extract audio data
                audio_chunk = model_output["tts_speech"]  # 音频段数据
                text_chunk = model_output["text_chunk"]  # 对应的字幕                

                start_time = ms_to_srt_time(audio_samples * 1000.0 / 22050)
                audio_samples += audio_chunk.shape[1]
                end_time = ms_to_srt_time(audio_samples * 1000.0 / 22050)
                new_subtitle = Subtitle.from_str_time(subtitle_index,start_time,end_time,text_chunk)
                new_subtitles.append(new_subtitle)
                logger.info(f'gererate text:{text_chunk}')

                audio_file = os.path.join(output_dir, f"{subtitle_index:03d}.wav")
                # Ensure audio_chunk is in the correct shape
                if audio_chunk.ndim == 1:  # If 1D (samples), add channel dimension
                    audio_chunk = audio_chunk.unsqueeze(0)

                torchaudio.save(audio_file, audio_chunk, 22050)
                logger.info(f"Saved audio file: {audio_file}")
                
                # Append the audio segment to the combined list
                combined_audio_segments.append(audio_chunk)
                
                subtitle_index += 1  
        return combined_audio_segments,new_subtitles
    
    def load_speaker(self,speaker):
        voice_path = f'{INPUT_DIR}/voices/{speaker}.pt'
        if(os.path.exists(voice_path)):
            return torch.load(voice_path, weights_only=True)
        else:
            logger.warning(f'load speaker error:{speaker} not exists!')
            return None
    
class CosyVoiceDubbingNode:
    def __init__(self):
        self.cosyvoice = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tts_srt": ("SRT",),
                "prompt_wav": ("AUDIO",),
                "language": (["<|zh|>", "<|en|>", "<|jp|>", "<|yue|>", "<|ko|>"],),
                "if_single": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "prompt_srt": ("SRT",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "ComfyUI-CosyVoice"

    def generate(self, tts_srt, prompt_wav, language, if_single, seed, prompt_srt=None):
        # Initialize the model
        model_dir = os.path.join(pretrained_models, "CosyVoice-300M")
        snapshot_download(model_id="iic/CosyVoice-300M", local_dir=model_dir)
        set_all_random_seed(seed)
        
        if self.cosyvoice is None:
            logger.info(f"Initializing CosyVoice model {model_dir} ...")
            self.cosyvoice = CosyVoice(model_dir)
        
        # Load subtitle files
        logger.debug("Loading TTS SRT subtitles...")
        text_subtitles = load_srt(tts_srt) #self.load_srt(tts_srt)
        prompt_subtitles = load_srt(prompt_srt) if prompt_srt else None

        # Preprocess prompt audio
        logger.debug("Preprocessing prompt audio...")
        speech_numpy, prompt_sr = self.preprocess_audio(prompt_wav)
        audio_seg = audiosegment.from_numpy_array(speech_numpy, prompt_sr)

        # Validate prompt audio duration
        assert audio_seg.duration_seconds > 3, "Prompt wav must be longer than 3 seconds."
        audio_seg.export(os.path.join(output_dir, "test.mp3"), format="mp3")
        
        # Generate synthesized audio segments
        new_audio_seg = audiosegment.silent(0, target_sr)  # target_sr = 22050

        for i, text_sub in enumerate(text_subtitles):
            logger.debug(f"Processing subtitle {i + 1}/{len(text_subtitles)}...")
            
            curr_tts_text, prompt_text_list, prompt_wav_seg = self.get_tts_text_and_prompt(
                text_sub, i, text_subtitles, language, if_single, audio_seg, prompt_srt, prompt_subtitles
            )
            prompt_wav_seg.export(os.path.join(output_dir,f"{i}_prompt.wav"),format="wav")

            # Convert prompt audio segment to Tensor
            prompt_wav_seg_numpy = prompt_wav_seg.to_numpy_array() / 32768  # Normalize to [-1, 1]
            # prompt_wav_seg_numpy = prompt_wav_seg_numpy.astype(np.float32)
            prompt_speech_16k = postprocess(torch.Tensor(prompt_wav_seg_numpy).unsqueeze(0))

            prompt_text = None
            if prompt_srt:                
                prompt_text = ','.join(prompt_text_list)

            # Run synthesis inference
            curr_output = self.run_inference(
                curr_tts_text, prompt_speech_16k, prompt_text, prompt_subtitles, i
            )

            # Align audio segment with subtitle timing
            tmp_audio = self.align_audio(curr_output, text_sub, text_subtitles, i, audio_seg)

            # Append the generated segment
            new_audio_seg += tmp_audio

        # Finalize and return the audio
        logger.debug("Finalizing the generated audio...")
        output_numpy = new_audio_seg.to_numpy_array() / 32768
        audio = {"waveform": torch.stack([torch.Tensor(output_numpy).unsqueeze(0)]), "sample_rate": target_sr}
        return (audio,)

    # def load_srt(self, srt_file):
    #     """Load and parse SRT subtitle file."""
    #     with open(srt_file, 'r', encoding="utf-8") as file:
    #         content = file.read()
    #     return list(SrtParse(content))

    def preprocess_audio(self, audio_data):
        """Preprocess prompt audio, including resampling and converting to NumPy format."""
        waveform = audio_data['waveform'].squeeze(0)
        source_sr = audio_data['sample_rate']
        speech = waveform.mean(dim=0, keepdim=True)

        if source_sr != 16000:
            logger.debug(f"Resampling audio from {source_sr}Hz to 16000Hz...")
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=16000)(speech)
        
        speech_numpy = (speech.squeeze(0).numpy() * 32768).astype(np.int16)
        return speech_numpy, 16000

    def get_tts_text_and_prompt(self, text_sub, index, subtitles, language, if_single, audio_seg, prompt_srt, prompt_subtitles):
        """Extract TTS text and prompt audio segment based on subtitle text and index."""
        start_time = text_sub.start.total_seconds() * 1000
        end_time = text_sub.end.total_seconds() * 1000

        curr_tts_text = language + text_sub.text if if_single else language + text_sub.text[1:]
        speaker_id = text_sub.text[0] if not if_single else None

        # Extract prompt audio segment
        prompt_wav_seg = audio_seg[start_time:end_time]
        prompt_text_list = []

        if prompt_srt:
            prompt_text_list.append(prompt_subtitles[index].text)

        # Extend prompt audio if duration is insufficient
        prompt_wav_seg = self.extend_prompt_wav(
            prompt_wav_seg, subtitles, index, speaker_id, if_single, audio_seg, prompt_srt, prompt_subtitles, prompt_text_list
        )

        return curr_tts_text, prompt_text_list,prompt_wav_seg

    def extend_prompt_wav(self, prompt_wav_seg, subtitles, index, speaker_id, if_single, audio_seg, prompt_srt, prompt_subtitles, prompt_text_list):
        """Extend prompt audio segment to ensure sufficient duration."""
        while prompt_wav_seg.duration_seconds < 30:
            # Extend forward
            for j in range(index + 1, len(subtitles)):
                added_segments = False
                j_start = subtitles[j].start.total_seconds() * 1000
                j_end = subtitles[j].end.total_seconds() * 1000
                if if_single or subtitles[j].text[0] == speaker_id:
                    prompt_wav_seg += audiosegment.silent(500, frame_rate=prompt_sr) + audio_seg[j_start:j_end]
                    added_segments = True  # 
                    if prompt_srt:
                        prompt_text_list.append(prompt_subtitles[j].text)

            # Extend backward
            for j in range(0, index):
                j_start = subtitles[j].start.total_seconds() * 1000
                j_end = subtitles[j].end.total_seconds() * 1000
                if if_single or subtitles[j].text[0] == speaker_id:
                    prompt_wav_seg += audiosegment.silent(500, frame_rate=prompt_sr) + audio_seg[j_start:j_end]
                    added_segments = True  # 标记添加成功
                    if prompt_srt:
                        prompt_text_list.append(prompt_subtitles[j].text)
            logger.debug(f"Current prompt_wav_seg duration: {prompt_wav_seg.duration_seconds} seconds")
            if prompt_wav_seg.duration_seconds > 3:
                break        
            if not added_segments:
                    logger.error("No segments added, breaking to avoid infinite loop")
                    break
        return prompt_wav_seg

    def run_inference(self, curr_tts_text, prompt_speech_16k, prompt_text, prompt_subtitles, index):
        """Run the speech synthesis model."""
        if prompt_text:            
            logger.info(f"Running inference with prompt text: {prompt_text}")
            return self.cosyvoice.inference_zero_shot(curr_tts_text, prompt_text, prompt_speech_16k)
        else:
            logger.info("Running cross-lingual inference...")
            return self.cosyvoice.inference_cross_lingual(curr_tts_text, prompt_speech_16k)

    def align_audio(self, curr_output, text_sub, subtitles, index, audio_seg):
        """Align generated audio with subtitle timing."""
        import types

        curr_output_numpy = None
        if isinstance(curr_output, types.GeneratorType):
            for value in curr_output:
                curr_output_numpy = (value['tts_speech'].squeeze(0).numpy() * 32768 ).astype(np.int16)
                break
        else:
            curr_output_numpy = (curr_output['tts_speech'].squeeze(0).numpy() * 32768).astype(np.int16)
        
        print(curr_output_numpy.shape)
        text_audio = audiosegment.from_numpy_array(curr_output_numpy, target_sr)

        start_time = text_sub.start.total_seconds() * 1000
        if index < len(subtitles) - 1:
            nxt_start = subtitles[index + 1].start.total_seconds() * 1000
            duration = nxt_start - start_time
        else:
            duration = audio_seg.duration_seconds * 1000 - start_time

        ratio = text_audio.duration_seconds * 1000 / duration
        if ratio > 1:
            tmp_audio = speed_change(curr_output_numpy, ratio, target_sr)
            tmp_audio = audiosegment.from_numpy_array(tmp_audio, target_sr)
        else:
            tmp_audio = text_audio + audiosegment.silent(duration - text_audio.duration_seconds * 1000, target_sr)

        return tmp_audio
    