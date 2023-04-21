import functools
import os
from typing import List, NamedTuple

import cv2
import imageio
import numpy as np
import torch
from flytekit import Resources, map_task, task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from gfpgan import GFPGANer

from ..utils.videoio import save_video_with_watermark
from .videoio import load_video_to_cv2


@task(requests=Resources(mem="8Gi", cpu="2"))
def restore(idx: int, images: List[np.ndarray], restorer: GFPGANer) -> np.ndarray:
    img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)

    # restore faces and background if necessary
    cropped_faces, restored_faces, r_img = restorer.enhance(
        img, has_aligned=False, only_center_face=False, paste_back=True
    )

    return cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)


enhancer_nt = NamedTuple(
    "enhancer_nt", images=List[np.ndarray], restorer=GFPGANer, indices=List[int]
)


@task(requests=Resources(mem="5Gi", cpu="3"))
def enhancer(
    images: str,
    method: str,
    bg_upsampler: str,
    save_dir: FlyteDirectory,
) -> enhancer_nt:
    save_dir_downloaded = save_dir.download()
    images_path = os.path.join(save_dir_downloaded, images)

    if os.path.isfile(images_path):  # handle video to images
        list_images = load_video_to_cv2(images_path)

    # ------------------------ set up GFPGAN restorer ------------------------
    if method == "gfpgan":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANv1.4"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    elif method == "RestoreFormer":
        arch = "RestoreFormer"
        channel_multiplier = 2
        model_name = "RestoreFormer"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    elif method == "codeformer":
        arch = "CodeFormer"
        channel_multiplier = 2
        model_name = "CodeFormer"
        url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    else:
        raise ValueError(f"Wrong model version {method}.")

    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler == "realesrgan":
        if not torch.cuda.is_available():  # CPU
            import warnings

            warnings.warn(
                "The unoptimized RealESRGAN is slow on CPU. We do not use it. "
                "If you really want to use it, please modify the corresponding codes."
            )
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path="/apdcephfs/private_shadowcun/SadTalker/gfpgan/weights/RealESRGAN_x2plus.pth",
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # determine model paths
    model_path = os.path.join("gfpgan/weights", model_name + ".pth")

    if not os.path.isfile(model_path):
        model_path = os.path.join("checkpoints", model_name + ".pth")

    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler,
    )

    return enhancer_nt(
        images=list_images,
        restorer=restorer,
        indices=list(range(len(list_images))),
    )


@task(requests=Resources(mem="10Gi", cpu="2"))
def enhancer_post_process(
    new_audio_path: str,
    restored_img: List[np.ndarray],
    save_dir: FlyteDirectory,
    video_name: str,
) -> FlyteFile:
    save_dir_downloaded = save_dir.download()
    video_name_enhancer = video_name + "_enhanced.mp4"
    enhanced_path = os.path.join(save_dir_downloaded, "temp_" + video_name_enhancer)
    av_path_enhancer = os.path.join(save_dir_downloaded, video_name_enhancer)
    return_path = av_path_enhancer

    imageio.mimsave(enhanced_path, restored_img, fps=float(25))

    save_video_with_watermark(
        enhanced_path,
        os.path.join(save_dir_downloaded, new_audio_path),
        av_path_enhancer,
        watermark=None,
    )
    print(f"The generated video is named {save_dir_downloaded}/{video_name_enhancer}")
    os.remove(enhanced_path)

    return FlyteFile(return_path)


@workflow
def enhancer_wf(
    images: str,
    method: str,
    bg_upsampler: str,
    video_name: str,
    save_dir: FlyteDirectory,
    new_audio_path: str,
) -> FlyteFile:
    enhancer_output = enhancer(
        images=images,
        method=method,
        bg_upsampler=bg_upsampler,
        save_dir=save_dir,
    )
    map_task_partial = functools.partial(
        restore, images=enhancer_output.images, restorer=enhancer_output.restorer
    )
    restored_img = map_task(map_task_partial)(idx=enhancer_output.indices)
    return enhancer_post_process(
        restored_img=restored_img,
        new_audio_path=new_audio_path,
        save_dir=save_dir,
        video_name=video_name,
    )
