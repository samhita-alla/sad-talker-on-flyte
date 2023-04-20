import os
import warnings

import cv2
import numpy as np
from skimage import img_as_ubyte

warnings.filterwarnings("ignore")
import json
from typing import List, NamedTuple, Optional

import imageio
import torch
from flytekit import Resources, conditional, task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from pydub import AudioSegment

from ..utils.face_enhancer import enhancer_wf
from ..utils.paste_pic import paste_pic
from ..utils.videoio import save_video_with_watermark
from .modules.make_animation import make_animation


@task
def noop_return_path(return_path: FlyteFile) -> FlyteFile:
    return return_path


animate_from_coeff_generate_nt = NamedTuple(
    "animate_from_coeff_generate_nt",
    full_video_path=str,
    return_path=FlyteFile,
    video_name=str,
    new_audio_path=str,
    save_dir=FlyteDirectory,
)


@task(requests=Resources(mem="10Gi", cpu="4"))
def animate_from_coeff_generate(
    frame_num: int,
    crop_info: str,
    video_save_dir: FlyteDirectory,
    audio_path: FlyteFile,
    pic_path: FlyteFile,
    video_name: str,
    predictions: List[torch.Tensor],
    preprocess: str,
) -> animate_from_coeff_generate_nt:
    predictions_video = torch.stack(predictions, dim=1)
    predictions_video = predictions_video.reshape((-1,) + predictions_video.shape[2:])
    predictions_video = predictions_video[:frame_num]

    video = []
    for idx in range(predictions_video.shape[0]):
        image = predictions_video[idx]
        image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
        video.append(image)
    result = img_as_ubyte(video)

    ### the generated video is 256x256, so we  keep the aspect ratio,
    original_size = json.loads(crop_info)[0]
    if original_size:
        result = [
            cv2.resize(
                result_i, (256, int(256.0 * original_size[1] / original_size[0]))
            )
            for result_i in result
        ]

    video_save_dir_downloaded = video_save_dir.download()
    vanilla_video_name = video_name + ".mp4"
    path = os.path.join(video_save_dir_downloaded, "temp_" + vanilla_video_name)
    imageio.mimsave(path, result, fps=float(25))

    av_path = os.path.join(video_save_dir_downloaded, vanilla_video_name)
    return_path = av_path

    audio_path_downloaded = audio_path.download()
    audio_name = os.path.splitext(os.path.split(audio_path_downloaded)[-1])[0]
    new_audio_path = os.path.join(video_save_dir_downloaded, audio_name + ".wav")
    start_time = 0
    sound = AudioSegment.from_mp3(audio_path_downloaded)
    frames = frame_num
    end_time = start_time + frames * 1 / 25 * 1000
    word1 = sound.set_frame_rate(16000)
    word = word1[start_time:end_time]
    word.export(new_audio_path, format="wav")

    save_video_with_watermark(path, new_audio_path, av_path, watermark=None)
    print(
        f"The generated video is named {vanilla_video_name} in {video_save_dir_downloaded}"
    )

    pic_path_downloaded = pic_path.download()
    if preprocess.lower() == "full":
        # only add watermark to the full image.
        video_name_full = video_name + "_full.mp4"
        full_video_path = os.path.join(video_save_dir_downloaded, video_name_full)
        return_path = full_video_path

        paste_pic(
            path,
            pic_path_downloaded,
            json.loads(crop_info),
            new_audio_path,
            full_video_path,
        )
        print(
            f"The generated video is named {video_save_dir_downloaded}/{video_name_full}"
        )
    else:
        full_video_path = av_path

    return animate_from_coeff_generate_nt(
        full_video_path=os.path.basename(full_video_path),
        return_path=FlyteFile(return_path),
        video_name=video_name,
        save_dir=FlyteDirectory(
            video_save_dir_downloaded, remote_directory=video_save_dir.remote_source
        ),
        new_audio_path=os.path.basename(new_audio_path),
    )


@workflow
def animate_from_coeff_generate_wf(
    free_view_checkpoint: FlyteFile,
    mapping_checkpoint: FlyteFile,
    config_path: str,
    device: str,
    video_save_dir: FlyteDirectory,
    pic_path: FlyteFile,
    crop_info: str,
    enhancer: str,
    source_image: torch.Tensor,
    source_semantics: torch.Tensor,
    frame_num: int,
    target_semantics_list: torch.Tensor,
    video_name: str,
    yaw_c_seq: torch.Tensor,
    pitch_c_seq: torch.Tensor,
    roll_c_seq: torch.Tensor,
    audio_path: FlyteFile,
    background_enhancer: str,
    preprocess: str,
) -> FlyteFile:
    predictions = make_animation(
        source_image=source_image,
        source_semantics=source_semantics,
        target_semantics=target_semantics_list,
        yaw_c_seq=yaw_c_seq,
        pitch_c_seq=pitch_c_seq,
        roll_c_seq=roll_c_seq,
        config_path=config_path,
        device=device,
        free_view_checkpoint=free_view_checkpoint,
        mapping_checkpoint=mapping_checkpoint,
    )
    animate_from_coeff_generate_output = animate_from_coeff_generate(
        frame_num=frame_num,
        crop_info=crop_info,
        video_save_dir=video_save_dir,
        audio_path=audio_path,
        pic_path=pic_path,
        video_name=video_name,
        predictions=predictions,
        preprocess=preprocess,
    )
    return (
        conditional("Face Enhancer")
        .if_(enhancer != "")
        .then(
            enhancer_wf(
                images=animate_from_coeff_generate_output.full_video_path,
                method=enhancer,
                bg_upsampler=background_enhancer,
                video_name=animate_from_coeff_generate_output.video_name,
                save_dir=video_save_dir,
                new_audio_path=animate_from_coeff_generate_output.new_audio_path,
            )
        )
        .else_()
        .then(
            noop_return_path(return_path=animate_from_coeff_generate_output.return_path)
        )
    )
