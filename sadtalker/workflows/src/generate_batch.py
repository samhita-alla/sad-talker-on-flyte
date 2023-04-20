import functools
import os
import random
from typing import List, NamedTuple, Optional

import numpy as np
import scipy.io as scio
import torch
from flytekit import Resources, map_task, task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from .utils import audio


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(
            wav, [0, audio_length - len(wav)], mode="constant", constant_values=0
        )
    return wav


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames


def generate_blink_seq(num_frames):
    ratio = np.zeros((num_frames, 1))
    frame_id = 0
    while frame_id in range(num_frames):
        start = 80
        if frame_id + start + 9 <= num_frames - 1:
            ratio[frame_id + start : frame_id + start + 9, 0] = [
                0.5,
                0.6,
                0.7,
                0.9,
                1,
                0.9,
                0.7,
                0.6,
                0.5,
            ]
            frame_id = frame_id + start + 9
        else:
            break
    return ratio


def generate_blink_seq_randomly(num_frames):
    ratio = np.zeros((num_frames, 1))
    if num_frames <= 20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        start = random.choice(range(min(10, num_frames), min(int(num_frames / 2), 70)))
        if frame_id + start + 5 <= num_frames - 1:
            ratio[frame_id + start : frame_id + start + 5, 0] = [
                0.5,
                0.9,
                1.0,
                0.9,
                0.5,
            ]
            frame_id = frame_id + start + 5
        else:
            break
    return ratio


@task(requests=Resources(mem="5Gi", cpu="1"))
def loop_get_data(i: int, orig_mel: np.ndarray, spec: np.ndarray) -> np.ndarray:
    fps = 25
    syncnet_mel_step_size = 16
    start_frame_num = i - 2
    start_idx = int(80.0 * (start_frame_num / float(fps)))
    end_idx = start_idx + syncnet_mel_step_size
    seq = np.arange(start_idx, end_idx)
    seq = np.clip(seq, 0, orig_mel.shape[0] - 1)
    m = spec[seq, :]
    return m.T


get_data_nt = NamedTuple(
    "get_data_nt",
    num_frames=int,
    range_frames=List[int],
    orig_mel=np.ndarray,
    spec=np.ndarray,
    audio_name=str,
)


@task(cache=True, cache_version="1.0")
def get_data(
    audio_path: FlyteFile,
) -> get_data_nt:
    audio_path_downloaded = audio_path.download()
    audio_name = os.path.splitext(os.path.split(audio_path_downloaded)[-1])[0]

    wav = audio.load_wav(audio_path_downloaded, 16000)
    wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()  # nframes 80

    return get_data_nt(
        num_frames=num_frames,
        range_frames=list(range(num_frames)),
        orig_mel=orig_mel,
        spec=spec,
        audio_name=audio_name,
    )


get_data_post_processing_nt = NamedTuple(
    "get_data_post_processing_nt",
    indiv_mels=torch.Tensor,
    ref_coeff=torch.Tensor,
    num_frames=int,
    ratio_gt=torch.Tensor,
    audio_name=str,
    pic_name=str,
)


@task(requests=Resources(mem="1Gi", cpu="2"))
def get_data_post_processing(
    indiv_mels: List[np.ndarray],
    device: str,
    ref_eyeblink_coeff_path: Optional[str],
    save_dir: FlyteDirectory,
    first_coeff_path: str,
    num_frames: int,
    audio_name: str,
    still: bool,
) -> get_data_post_processing_nt:
    save_dir_path = save_dir.download()
    first_coeff_path_downloaded = os.path.join(save_dir_path, first_coeff_path)
    if first_coeff_path_downloaded is None:
        print("Can't get the coeffs of the input")
        return

    pic_name = os.path.splitext(os.path.split(first_coeff_path_downloaded)[-1])[0]
    indiv_mels = np.asarray(indiv_mels)  # T 80 16

    ratio = generate_blink_seq_randomly(num_frames)  # T
    source_semantics_path = first_coeff_path_downloaded
    source_semantics_dict = scio.loadmat(source_semantics_path)
    ref_coeff = source_semantics_dict["coeff_3dmm"][:1, :70]  # 1 70
    ref_coeff = np.repeat(ref_coeff, num_frames, axis=0)

    if ref_eyeblink_coeff_path is not None:
        ratio[:num_frames] = 0
        refeyeblink_coeff_dict = scio.loadmat(
            os.path.join(save_dir_path, ref_eyeblink_coeff_path)
        )
        refeyeblink_coeff = refeyeblink_coeff_dict["coeff_3dmm"][:, :64]
        refeyeblink_num_frames = refeyeblink_coeff.shape[0]
        if refeyeblink_num_frames < num_frames:
            div = num_frames // refeyeblink_num_frames
            re = num_frames % refeyeblink_num_frames
            refeyeblink_coeff_list = [refeyeblink_coeff for i in range(div)]
            refeyeblink_coeff_list.append(refeyeblink_coeff[:re, :64])
            refeyeblink_coeff = np.concatenate(refeyeblink_coeff_list, axis=0)

        ref_coeff[:, :64] = refeyeblink_coeff[:num_frames, :64]

    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0)  # bs T 1 80 16
    if still:
        ratio = torch.FloatTensor(ratio).unsqueeze(0).fill_(0.0)  # bs T
    else:
        ratio = torch.FloatTensor(ratio).unsqueeze(0)
    ref_coeff = torch.FloatTensor(ref_coeff).unsqueeze(0)  # bs 1 70

    indiv_mels = indiv_mels.to(device)
    ratio = ratio.to(device)
    ref_coeff = ref_coeff.to(device)

    return get_data_post_processing_nt(
        indiv_mels=indiv_mels,
        ref_coeff=ref_coeff,
        num_frames=num_frames,
        ratio_gt=ratio,
        audio_name=audio_name,
        pic_name=pic_name,
    )


@workflow
def generate_batch_wf(
    save_dir: FlyteDirectory,
    first_coeff_path: str,
    audio_path: FlyteFile,
    device: str,
    ref_eyeblink_coeff_path: Optional[str],
    still: bool,
) -> get_data_post_processing_nt:
    get_data_output = get_data(audio_path=audio_path)
    map_task_partial = functools.partial(
        loop_get_data,
        orig_mel=get_data_output.orig_mel,
        spec=get_data_output.spec,
    )
    indiv_mels = map_task(map_task_partial)(i=get_data_output.range_frames)
    return get_data_post_processing(
        indiv_mels=indiv_mels,
        device=device,
        ref_eyeblink_coeff_path=ref_eyeblink_coeff_path,
        save_dir=save_dir,
        first_coeff_path=first_coeff_path,
        num_frames=get_data_output.num_frames,
        audio_name=get_data_output.audio_name,
        still=still,
    )
