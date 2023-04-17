# check the sync of 3dmm feature and the audio
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import scipy.io as scio
import torch
from dataclasses_json import dataclass_json
from flytekit import task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from tqdm import tqdm

from .models.facerecon_model import FaceReconModel


@dataclass_json
@dataclass
class ModelParams:
    driven_audio: FlyteFile = "https://huggingface.co/spaces/vinthony/SadTalker/raw/main/examples/driven_audio/bus_chinese.wav"
    source_image: FlyteFile = "https://huggingface.co/spaces/vinthony/SadTalker/raw/main/examples/source_image/full_body_2.png"
    ref_pose: Optional[str] = None
    ref_eyeblink: Optional[str] = None
    result_dir: str = "results"
    pose_style: int = 0
    batch_size: int = 2
    expression_scale: float = 1.0
    input_yaw: List[int] = field(default_factory=lambda: [0])
    input_pitch: List[int] = field(default_factory=lambda: [0])
    input_roll: List[int] = field(default_factory=lambda: [0])
    enhancer: Optional[str] = None
    background_enhancer: Optional[str] = None
    device: str = "cpu"
    face3dvis: bool = False
    still: bool = True
    preprocess: str = "crop"
    net_recon: str = "resnet50"
    full_img_enhancer: Optional[str] = None
    use_last_fc: bool = False
    focal: float = 1015.0
    center: float = 112.0
    camera_d: float = 10.0
    z_near: float = 5.0
    z_far: float = 15.0
    bfm_folder: str = "./checkpoints/BFM_Fitting/"
    bfm_model: str = "BFM_model_front.mat"
    checkpoint_dir: str = "./checkpoints"


# draft
@task
def gen_composed_video(
    args: ModelParams,
    device: str,
    first_frame_coeff: str,
    save_dir: FlyteDirectory,
    coeff_path: str,
    audio_path: FlyteFile,
    save_path: str,
) -> FlyteDirectory:
    save_dir_downloaded = save_dir.download()
    coeff_first = scio.loadmat(os.path.join(save_dir_downloaded, first_frame_coeff))[
        "full_3dmm"
    ]

    coeff_pred = scio.loadmat(os.path.join(save_dir_downloaded, coeff_path))[
        "coeff_3dmm"
    ]

    coeff_full = np.repeat(coeff_first, coeff_pred.shape[0], axis=0)  # 257

    coeff_full[:, 80:144] = coeff_pred[:, 0:64]
    coeff_full[:, 224:227] = coeff_pred[:, 64:67]  # 3 dim translation
    coeff_full[:, 254:] = coeff_pred[:, 67:]  # 3 dim translation

    tmp_video_path = "/tmp/face3dtmp.mp4"

    facemodel = FaceReconModel(args)

    video = cv2.VideoWriter(
        tmp_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (224, 224)
    )

    for k in tqdm(range(coeff_pred.shape[0]), "face3d rendering:"):
        cur_coeff_full = torch.tensor(coeff_full[k : k + 1], device=device)

        facemodel.forward(cur_coeff_full, device)

        predicted_landmark = facemodel.pred_lm  # TODO.
        predicted_landmark = predicted_landmark.cpu().numpy().squeeze()

        rendered_img = facemodel.pred_face
        rendered_img = 255.0 * rendered_img.cpu().numpy().squeeze().transpose(1, 2, 0)
        out_img = rendered_img[:, :, :3].astype(np.uint8)

        video.write(np.uint8(out_img[:, :, ::-1]))

    video.release()

    save_path = os.path.join(save_dir_downloaded, save_path)

    command = "ffmpeg -v quiet -y -i {} -i {} -strict -2 -q:v 1 {}".format(
        audio_path.download(), tmp_video_path, save_path
    )
    subprocess.call(command, shell=platform.system() != "Windows")

    return FlyteDirectory(save_dir_downloaded, remote_directory=save_dir.remote_source)
