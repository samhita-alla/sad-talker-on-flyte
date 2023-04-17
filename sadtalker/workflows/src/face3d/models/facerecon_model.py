"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import torch
import trimesh
from scipy.io import savemat

from ..util import util
from ..util.nvdiffrast import MeshRenderer
from .base_model import BaseModel
from .bfm import ParametricFaceModel
from .losses import (
    landmark_loss,
    perceptual_loss,
    photo_loss,
    reflectance_loss,
    reg_loss,
)
from .networks import define_net_recog


class FaceReconModel(BaseModel):
    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        self.visual_names = ["output_vis"]
        self.model_names = ["net_recon"]
        self.parallel_names = self.model_names + ["renderer"]

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder,
            camera_distance=opt.camera_d,
            focal=opt.focal,
            center=opt.center,
            is_train=self.isTrain,
            default_name=opt.bfm_model,
        )

        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov,
            znear=opt.z_near,
            zfar=opt.z_far,
            rasterize_size=int(2 * opt.center),
        )

        if self.isTrain:
            self.loss_names = ["all", "feat", "color", "lm", "reg", "gamma", "reflc"]

            self.net_recog = define_net_recog(
                net_recog=opt.net_recog, pretrained_path=opt.net_recog_path
            )
            # loss func name: (compute_%s_loss) % loss_name
            self.compute_feat_loss = perceptual_loss
            self.comupte_color_loss = photo_loss
            self.compute_lm_loss = landmark_loss
            self.compute_reg_loss = reg_loss
            self.compute_reflc_loss = reflectance_loss

            self.optimizer = torch.optim.Adam(self.net_recon.parameters(), lr=opt.lr)
            self.optimizers = [self.optimizer]
            self.parallel_names += ["net_recog"]
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input["imgs"].to(self.device)
        self.atten_mask = input["msks"].to(self.device) if "msks" in input else None
        self.gt_lm = input["lms"].to(self.device) if "lms" in input else None
        self.trans_m = input["M"].to(self.device) if "M" in input else None
        self.image_paths = input["im_paths"] if "im_paths" in input else None

    def forward(self, output_coeff, device):
        self.facemodel.to(device)
        (
            self.pred_vertex,
            self.pred_tex,
            self.pred_color,
            self.pred_lm,
        ) = self.facemodel.compute_for_render(output_coeff)
        self.pred_mask, _, self.pred_face = self.renderer(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color
        )

        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)

    def compute_losses(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        assert self.net_recog.training == False
        trans_m = self.trans_m
        if not self.opt.use_predef_M:
            trans_m = estimate_norm_torch(self.pred_lm, self.input_img.shape[-2])

        pred_feat = self.net_recog(self.pred_face, trans_m)
        gt_feat = self.net_recog(self.input_img, self.trans_m)
        self.loss_feat = self.opt.w_feat * self.compute_feat_loss(pred_feat, gt_feat)

        face_mask = self.pred_mask
        if self.opt.use_crop_face:
            face_mask, _, _ = self.renderer(
                self.pred_vertex, self.facemodel.front_face_buf
            )

        face_mask = face_mask.detach()
        self.loss_color = self.opt.w_color * self.comupte_color_loss(
            self.pred_face, self.input_img, self.atten_mask * face_mask
        )

        loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict, self.opt)
        self.loss_reg = self.opt.w_reg * loss_reg
        self.loss_gamma = self.opt.w_gamma * loss_gamma

        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(self.pred_lm, self.gt_lm)

        self.loss_reflc = self.opt.w_reflc * self.compute_reflc_loss(
            self.pred_tex, self.facemodel.skin_mask
        )

        self.loss_all = (
            self.loss_feat
            + self.loss_color
            + self.loss_reg
            + self.loss_gamma
            + self.loss_lm
            + self.loss_reflc
        )

    def optimize_parameters(self, isTrain=True):
        self.forward()
        self.compute_losses()
        """Update network weights; it will be called in every training iteration."""
        if isTrain:
            self.optimizer.zero_grad()
            self.loss_all.backward()
            self.optimizer.step()

    def compute_visuals(self):
        with torch.no_grad():
            input_img_numpy = (
                255.0 * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            )
            output_vis = (
                self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            )
            output_vis_numpy_raw = (
                255.0 * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            )

            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_numpy = util.draw_landmarks(
                    output_vis_numpy_raw, gt_lm_numpy, "b"
                )
                output_vis_numpy = util.draw_landmarks(
                    output_vis_numpy, pred_lm_numpy, "r"
                )

                output_vis_numpy = np.concatenate(
                    (input_img_numpy, output_vis_numpy_raw, output_vis_numpy), axis=-2
                )
            else:
                output_vis_numpy = np.concatenate(
                    (input_img_numpy, output_vis_numpy_raw), axis=-2
                )

            self.output_vis = (
                torch.tensor(output_vis_numpy / 255.0, dtype=torch.float32)
                .permute(0, 3, 1, 2)
                .to(self.device)
            )

    def save_mesh(self, name):
        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = (
            10 - recon_shape[..., -1]
        )  # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.facemodel.face_buf.cpu().numpy()
        mesh = trimesh.Trimesh(
            vertices=recon_shape,
            faces=tri,
            vertex_colors=np.clip(255.0 * recon_color, 0, 255).astype(np.uint8),
        )
        mesh.export(name)

    def save_coeff(self, name):
        pred_coeffs = {
            key: self.pred_coeffs_dict[key].cpu().numpy()
            for key in self.pred_coeffs_dict
        }
        pred_lm = self.pred_lm.cpu().numpy()
        pred_lm = np.stack(
            [pred_lm[:, :, 0], self.input_img.shape[2] - 1 - pred_lm[:, :, 1]], axis=2
        )  # transfer to image coordinate
        pred_coeffs["lm68"] = pred_lm
        savemat(name, pred_coeffs)
