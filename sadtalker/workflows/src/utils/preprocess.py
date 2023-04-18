import functools
import json
import os
from typing import List, NamedTuple
from zipfile import ZipFile

import cv2
import flytekit
import numpy as np
import torch
from flytekit import Resources, map_task, task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from PIL import Image
from scipy.io import savemat

from ..face3d.extract_kp_videos import KeypointExtractor
from ..face3d.models.networks import define_net_recon
from ..face3d.util.load_mats import load_lm3d
from ..face3d.util.preprocess import align_img
from .croper import Croper


def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors
    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80:144]
    tex_coeffs = coeffs[:, 144:224]
    angles = coeffs[:, 224:227]
    gammas = coeffs[:, 227:254]
    translations = coeffs[:, 254:]
    return {
        "id": id_coeffs,
        "exp": exp_coeffs,
        "tex": tex_coeffs,
        "angle": angles,
        "gamma": gammas,
        "trans": translations,
    }


@task(requests=Resources(mem="5Gi", cpu="2"))
def loop_frames_pil(
    idx: int,
    frames_pil: List[Image.Image],
    lm: np.ndarray,
    device: str,
    lm3d_std: np.ndarray,
    net_recon: torch.nn.Module,
) -> List[np.ndarray]:
    frame = frames_pil[idx]
    W, H = frame.size
    lm1 = lm[idx].reshape([-1, 2])

    if np.mean(lm1) == -1:
        lm1 = (lm3d_std[:, :2] + 1) / 2.0
        lm1 = np.concatenate([lm1[:, :1] * W, lm1[:, 1:2] * H], 1)
    else:
        lm1[:, -1] = H - 1 - lm1[:, -1]

    trans_params, im1, lm1, _ = align_img(frame, lm1, lm3d_std)

    trans_params = np.array(
        [float(item) for item in np.hsplit(trans_params, 5)]
    ).astype(np.float32)
    im_t = (
        torch.tensor(np.array(im1) / 255.0, dtype=torch.float32)
        .permute(2, 0, 1)
        .to(device)
        .unsqueeze(0)
    )

    with torch.no_grad():
        full_coeff = net_recon(im_t)
        coeffs = split_coeff(full_coeff)

    pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}

    pred_coeff = np.concatenate(
        [
            pred_coeff["exp"],
            pred_coeff["angle"],
            pred_coeff["trans"],
            trans_params[2:][None],
        ],
        1,
    )
    return [pred_coeff, full_coeff.cpu().numpy()]


crop_and_extract_nt = NamedTuple(
    "crop_and_extract_nt",
    list_frames_pil=List[int],
    crop_info=str,
    frames_pil=List[Image.Image],
    lm=np.ndarray,
    lm3d_std=np.ndarray,
    net_recon=torch.nn.Module,
    save_dir=FlyteDirectory,
    coeff_path=str,
    png_path=str,
)


@task(requests=Resources(mem="5Gi", cpu="3"))
def crop_and_extract(
    path_of_lm_croper: FlyteFile,
    path_of_net_recon_model: FlyteFile,
    dir_of_BFM_fitting: FlyteFile,
    device: str,
    input_path: FlyteFile,
    save_dir: FlyteDirectory,
    save_dir_name: str,
    crop_or_resize: str,
    source_image_flag: bool,
) -> crop_and_extract_nt:
    croper = Croper(path_of_lm_croper.download())
    kp_extractor = KeypointExtractor(device)
    net_recon = define_net_recon(
        net_recon="resnet50", use_last_fc=False, init_path=""
    ).to(device)
    checkpoint = torch.load(
        path_of_net_recon_model.download(), map_location=torch.device(device)
    )
    net_recon.load_state_dict(checkpoint["net_recon"])
    net_recon.eval()

    working_dir = flytekit.current_context().working_directory

    zip_file = dir_of_BFM_fitting.download()
    with ZipFile(zip_file) as z_obj:
        z_obj.extractall(path=working_dir)

    lm3d_std = load_lm3d(os.path.join(working_dir, "BFM_Fitting"))

    pic_size = 256
    input_path_downloaded = input_path.download()
    pic_name = os.path.splitext(os.path.split(input_path_downloaded)[-1])[0]

    save_dir_path = save_dir.download()
    frame_dir = os.path.join(save_dir_path, save_dir_name)
    os.makedirs(frame_dir, exist_ok=True)

    landmarks_path = os.path.join(frame_dir, pic_name + "_landmarks.txt")
    coeff_path = os.path.join(save_dir_name, pic_name + ".mat")
    png_path = os.path.join(save_dir_name, pic_name + ".png")

    # load input
    if not os.path.isfile(input_path_downloaded):
        raise ValueError("input_path must be a valid path to video/image file")
    elif input_path_downloaded.split(".")[1] in ["jpg", "png", "jpeg"]:
        # loader for first frame
        full_frames = [cv2.imread(input_path_downloaded)]
        fps = 25
    else:
        # loader for videos
        video_stream = cv2.VideoCapture(input_path_downloaded)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            full_frames.append(frame)
            if source_image_flag:
                break

    x_full_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames]

    if crop_or_resize.lower() in ["crop", "full"]:  # default crop
        x_full_frames, crop, quad = croper.crop(x_full_frames, still=True, xsize=512)
        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx
        crop_info = json.dumps([(ox2 - ox1, oy2 - oy1), list(crop), list(quad)])
    else:
        oy1, oy2, ox1, ox2 = (
            0,
            x_full_frames[0].shape[0],
            0,
            x_full_frames[0].shape[1],
        )
        crop_info = json.dumps([(ox2 - ox1, oy2 - oy1), None, None])

    frames_pil = [
        Image.fromarray(cv2.resize(frame, (pic_size, pic_size)))
        for frame in x_full_frames
    ]
    if len(frames_pil) == 0:
        print("No face is detected in the input file")
        return None, None

    # save crop info
    for frame in frames_pil:
        cv2.imwrite(
            os.path.join(save_dir_path, png_path),
            cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR),
        )

    # 2. get the landmark according to the detected face.
    if not os.path.isfile(landmarks_path):
        lm = kp_extractor.extract_keypoint(frames_pil, landmarks_path)
    else:
        print("Using saved landmarks.")
        lm = np.loadtxt(landmarks_path).astype(np.float32)
        lm = lm.reshape([len(x_full_frames), -1, 2])

    return crop_and_extract_nt(
        list_frames_pil=list(range(len(frames_pil))),
        crop_info=crop_info,
        frames_pil=frames_pil,
        lm=lm,
        lm3d_std=lm3d_std,
        net_recon=net_recon,
        save_dir=FlyteDirectory(save_dir_path, remote_directory=save_dir.remote_source),
        coeff_path=coeff_path,
        png_path=png_path,
    )


@task(requests=Resources(mem="500Mi", cpu="2"))
def save_mat_task(
    list_loop_frames_pil_output: List[List[np.ndarray]],
    save_dir: FlyteDirectory,
    coeff_path: str,
) -> FlyteDirectory:
    video_coeffs = [x[0] for x in list_loop_frames_pil_output]
    full_coeffs = [x[1] for x in list_loop_frames_pil_output]
    semantic_npy = np.array(video_coeffs)[:, 0]
    save_dir_path = save_dir.download()
    savemat(
        os.path.join(save_dir_path, coeff_path),
        {"coeff_3dmm": semantic_npy, "full_3dmm": np.array(full_coeffs)[0]},
    )
    return FlyteDirectory(save_dir_path, remote_directory=save_dir.remote_source)


crop_and_extract_wf_nt = NamedTuple(
    "crop_and_extract_wf_nt",
    coeff_path=str,
    png_path=str,
    crop_info=str,
)


@workflow
def crop_and_extract_wf(
    path_of_lm_croper: FlyteFile,
    path_of_net_recon_model: FlyteFile,
    dir_of_BFM_fitting: FlyteFile,
    device: str,
    input_path: FlyteFile,
    save_dir: FlyteDirectory,
    save_dir_name: str,
    crop_or_resize: str = "crop",
    source_image_flag: bool = False,
) -> crop_and_extract_wf_nt:
    crop_and_extract_output = crop_and_extract(
        path_of_lm_croper=path_of_lm_croper,
        path_of_net_recon_model=path_of_net_recon_model,
        dir_of_BFM_fitting=dir_of_BFM_fitting,
        device=device,
        input_path=input_path,
        save_dir=save_dir,
        save_dir_name=save_dir_name,
        crop_or_resize=crop_or_resize,
        source_image_flag=source_image_flag,
    )
    map_task_partial = functools.partial(
        loop_frames_pil,
        frames_pil=crop_and_extract_output.frames_pil,
        lm=crop_and_extract_output.lm,
        device=device,
        lm3d_std=crop_and_extract_output.lm3d_std,
        net_recon=crop_and_extract_output.net_recon,
    )
    list_loop_frames_pil_output = map_task(map_task_partial)(
        idx=crop_and_extract_output.list_frames_pil
    )
    save_mat_task(
        list_loop_frames_pil_output=list_loop_frames_pil_output,
        save_dir=save_dir,
        coeff_path=crop_and_extract_output.coeff_path,
    )
    return crop_and_extract_wf_nt(
        coeff_path=crop_and_extract_output.coeff_path,
        png_path=crop_and_extract_output.png_path,
        crop_info=crop_and_extract_output.crop_info,
    )
