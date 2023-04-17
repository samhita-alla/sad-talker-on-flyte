import io
import os
from typing import NamedTuple, Optional

import numpy as np
import torch
import urllib3
from flytekit import Resources, task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from scipy.io import loadmat, savemat
from scipy.signal import savgol_filter
from yacs.config import CfgNode as CN

from .audio2exp_models.audio2exp import Audio2Exp
from .audio2exp_models.networks import SimpleWrapperV2
from .audio2pose_models.audio2pose import Audio2Pose


def load_cpk(checkpoint_path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["epoch"]


class NamedTextIOWrapper(io.TextIOWrapper):
    def __init__(self, buffer, name=None, **kwargs):
        vars(self)["name"] = name
        super().__init__(buffer, **kwargs)

    def __getattribute__(self, name):
        if name == "name":
            return vars(self)["name"]
        return super().__getattribute__(name)


def using_refpose(coeffs_pred_numpy, ref_pose_coeff_path):
    num_frames = coeffs_pred_numpy.shape[0]
    refpose_coeff_dict = loadmat(ref_pose_coeff_path)
    refpose_coeff = refpose_coeff_dict["coeff_3dmm"][:, 64:70]
    refpose_num_frames = refpose_coeff.shape[0]
    if refpose_num_frames < num_frames:
        div = num_frames // refpose_num_frames
        re = num_frames % refpose_num_frames
        refpose_coeff_list = [refpose_coeff for i in range(div)]
        refpose_coeff_list.append(refpose_coeff[:re, :])
        refpose_coeff = np.concatenate(refpose_coeff_list, axis=0)

    coeffs_pred_numpy[:, 64:70] = refpose_coeff[:num_frames, :]
    return coeffs_pred_numpy


generate_nt = NamedTuple("generate_nt", save_dir=FlyteDirectory, coeff_path=str)


@task(requests=Resources(mem="1Gi", cpu="2"))
def generate_audio_to_coeff(
    audio2pose_checkpoint: FlyteFile,
    audio2pose_yaml_path: str,
    audio2exp_checkpoint: FlyteFile,
    audio2exp_yaml_path: str,
    wav2lip_checkpoint: FlyteFile,
    device: str,
    coeff_save_dir: FlyteDirectory,
    pose_style: int,
    ref_pose_coeff_path: Optional[str],
    indiv_mels: torch.Tensor,
    ref: torch.Tensor,
    num_frames: int,
    ratio_get: torch.Tensor,
    audio_name: str,
    pic_name: str,
) -> generate_nt:
    # load config
    http = urllib3.PoolManager()
    pose_response = http.request("GET", audio2pose_yaml_path, preload_content=False)
    pose_response.auto_close = False
    fcfg_pose = NamedTextIOWrapper(pose_response, name="audio2pose.yaml")
    cfg_pose = CN.load_cfg(fcfg_pose)
    cfg_pose.freeze()

    exp_response = http.request("GET", audio2exp_yaml_path, preload_content=False)
    exp_response.auto_close = False
    fcfg_exp = NamedTextIOWrapper(exp_response, name="audio2exp.yaml")
    cfg_exp = CN.load_cfg(fcfg_exp)
    cfg_exp.freeze()

    # load audio2pose_model
    audio2pose_model = Audio2Pose(
        cfg_pose, wav2lip_checkpoint.download(), device=device
    )
    audio2pose_model = audio2pose_model.to(device)
    audio2pose_model.eval()
    for param in audio2pose_model.parameters():
        param.requires_grad = False
    try:
        load_cpk(
            audio2pose_checkpoint.download(), model=audio2pose_model, device=device
        )
    except:
        raise Exception("Failed in loading audio2pose_checkpoint")

    # load audio2exp_model
    netG = SimpleWrapperV2()
    netG = netG.to(device)
    for param in netG.parameters():
        netG.requires_grad = False
    netG.eval()
    try:
        load_cpk(audio2exp_checkpoint.download(), model=netG, device=device)
    except:
        raise Exception("Failed in loading audio2exp_checkpoint")
    audio2exp_model = Audio2Exp(
        netG, cfg_exp, device=device, prepare_training_loss=False
    )
    audio2exp_model = audio2exp_model.to(device)
    for param in audio2exp_model.parameters():
        param.requires_grad = False
    audio2exp_model.eval()

    with torch.no_grad():
        # test
        results_dict_exp = audio2exp_model.test(
            indiv_mels,
            ref,
            ratio_get,
        )
        exp_pred = results_dict_exp["exp_coeff_pred"]  # bs T 64

        batch_class = torch.LongTensor([pose_style]).to(device)
        results_dict_pose = audio2pose_model.test(
            ref, batch_class, indiv_mels, num_frames
        )
        pose_pred = results_dict_pose["pose_pred"]  # bs T 6

        pose_len = pose_pred.shape[1]
        if pose_len < 13:
            pose_len = int((pose_len - 1) / 2) * 2 + 1
            pose_pred = torch.Tensor(
                savgol_filter(np.array(pose_pred.cpu()), pose_len, 2, axis=1)
            ).to(device)
        else:
            pose_pred = torch.Tensor(
                savgol_filter(np.array(pose_pred.cpu()), 13, 2, axis=1)
            ).to(device)

        coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)  # bs T 70

        coeffs_pred_numpy = coeffs_pred[0].clone().detach().cpu().numpy()

        coeff_save_dir_downloaded = coeff_save_dir.download()
        if ref_pose_coeff_path is not None:
            coeffs_pred_numpy = using_refpose(
                coeffs_pred_numpy,
                os.path.join(coeff_save_dir_downloaded, ref_pose_coeff_path),
            )

        coeff_path = "%s##%s.mat" % (pic_name, audio_name)
        savemat(
            os.path.join(
                coeff_save_dir_downloaded,
                coeff_path,
            ),
            {"coeff_3dmm": coeffs_pred_numpy},
        )

        return generate_nt(
            save_dir=FlyteDirectory(
                coeff_save_dir_downloaded, remote_directory=coeff_save_dir.remote_source
            ),
            coeff_path=coeff_path,
        )


@workflow
def audio_to_coeff_wf(
    audio2pose_checkpoint: FlyteFile,
    audio2pose_yaml_path: str,
    audio2exp_checkpoint: FlyteFile,
    audio2exp_yaml_path: str,
    wav2lip_checkpoint: FlyteFile,
    device: str,
    save_dir: FlyteDirectory,
    pose_style: int,
    ref_pose_coeff_path: Optional[str],
    indiv_mels: torch.Tensor,
    ref_coeff: torch.Tensor,
    num_frames: int,
    ratio_get: torch.Tensor,
    audio_name: str,
    pic_name: str,
) -> str:
    return generate_audio_to_coeff(
        ref_pose_coeff_path=ref_pose_coeff_path,
        audio2pose_checkpoint=audio2pose_checkpoint,
        audio2pose_yaml_path=audio2pose_yaml_path,
        audio2exp_checkpoint=audio2exp_checkpoint,
        audio2exp_yaml_path=audio2exp_yaml_path,
        wav2lip_checkpoint=wav2lip_checkpoint,
        device=device,
        coeff_save_dir=save_dir,
        pose_style=pose_style,
        audio_name=audio_name,
        pic_name=pic_name,
        indiv_mels=indiv_mels,
        ref=ref_coeff,
        num_frames=num_frames,
        ratio_get=ratio_get,
    ).coeff_path
