import os
import sys
import argparse
from random import seed
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.BaseDataset import build_dataset
from src.models.fusion import Fusion
from src.config import save_config
from src.utils.data import to_device
from src.utils.io import write_pfm, write_cam_sfm, display_map
from src.loss import compute_loss

class VFUSE():
    def __init__(self, cfg, training_scenes=None, validation_scenes=None, inference_scene=None):
        self.cfg = cfg
        self.device = self.cfg["device"]
        self.mode = self.cfg["mode"]
        self.training_scenes = training_scenes
        self.validation_scenes = validation_scenes
        self.inference_scene = inference_scene

        # build the data loaders
        self.build_dataset()

        # build model
        if (self.mode == "training"):
            self.depth_planes = self.cfg["training"]["depth_planes"]
        else:
            self.depth_planes = self.cfg["eval"]["depth_planes"]
        self.model = Fusion(self.cfg, self.depth_planes).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params}")
        print(f"Trainable Model parameters: {train_params}")
        if (self.mode != "training"):
            self.model.load_state_dict(torch.load(self.cfg["model"])["model_state_dict"])

        self.log_path = self.cfg["log_path"]
        self.ckpt_path = os.path.join(self.log_path, "ckpts")
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)

        if (self.mode == "training"):
            self.final_ckpt_file = os.path.join(self.ckpt_path, "model.pt") 

            if (self.cfg["training"]["ckpt_file"] != None):
                print(f"Training model from pretrained check-point: {self.cfg['training']['ckpt_file']}")
                self.model.load_state_dict(torch.load(self.cfg['training']["ckpt_file"])["model_state_dict"])

            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.cfg["training"]["base_lr"])
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                                                        optimizer=self.optimizer,
                                                        step_size=self.cfg["training"]["step_size"],
                                                        gamma=self.cfg["training"]["gamma"])

        elif (self.mode=="inference"):
            # set data paths
            self.data_path = os.path.join(self.cfg["data_path"], self.inference_scene[0])
            self.output_path = os.path.join(self.cfg["output_path"], self.inference_scene[0])
            self.image_path = os.path.join(self.output_path, "images")
            self.cam_path = os.path.join(self.output_path, "cams")
            self.input_depth_path = os.path.join(self.output_path, "depths_input")
            self.input_conf_path = os.path.join(self.output_path, "confs_input")
            self.fused_depth_path = os.path.join(self.output_path, "depths_fused")
            self.fused_conf_path = os.path.join(self.output_path, "confs_fused")

            # create paths if they dont exist
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(self.image_path, exist_ok=True)
            os.makedirs(self.cam_path, exist_ok=True)
            os.makedirs(self.input_depth_path, exist_ok=True)
            os.makedirs(self.input_conf_path, exist_ok=True)
            os.makedirs(self.fused_depth_path, exist_ok=True)
            os.makedirs(self.fused_conf_path, exist_ok=True)
        else:
            print(f"Unknown operation mode '{self.mode}'")
            sys.exit()

        # log current configuration used
        save_config(self.log_path, self.cfg)

    def build_dataset(self):
        if (self.mode=="training"):
            self.train_dataset = build_dataset(self.cfg, self.mode, self.training_scenes)
            self.train_data_loader = DataLoader(self.train_dataset,
                                         self.cfg["training"]["batch_size"],
                                         shuffle=True,
                                         num_workers=self.cfg["training"]["num_workers"],
                                         pin_memory=True,
                                         drop_last=True)

            self.val_dataset = build_dataset(self.cfg, "validation", self.validation_scenes)
            self.val_data_loader = DataLoader(self.val_dataset,
                                         1,
                                         shuffle=False,
                                         num_workers=self.cfg["training"]["num_workers"],
                                         pin_memory=True,
                                         drop_last=False)
            train_sample_size = self.train_data_loader.__len__()
            val_sample_size = self.val_data_loader.__len__()
            print ('Training sample number: ', train_sample_size)
            print ('Validation sample number: ', val_sample_size)

        else:
            self.dataset = build_dataset(self.cfg, self.mode, self.inference_scene)
            self.data_loader = DataLoader(self.dataset,
                                         1,
                                         shuffle=False,
                                         num_workers=self.cfg["training"]["num_workers"],
                                         pin_memory=False,
                                         drop_last=False)



    def training(self):
        for epoch in range(self.cfg["training"]["epochs"]):
            self.train_step(epoch)
            self.val_step(epoch)

            # update learning rate
            scheduler.step()

            # save the model checkpoint
            if (epoch % self.cfg["training"]["ckpt_freq"]):
                ckpt_file = os.path.join(self.ckpt_path, f"model_{epoch:04d}.pt")
                torch.save({
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict()
                            }, ckpt_file)

        # save the final model
        torch.save({"model_state_dict":self.model.state_dict()}, self.final_ckpt_file)

    def train_step(self, epoch):
        self.model.train()
        loss_sum = {
                "loss": 0.0,
                "depth": 0.0,
                "coverage": 0.0,
                "radius": 0.0,
                "one_acc": 0.0,
                "three_acc": 0.0
                }

        with tqdm(self.train_data_loader, desc=f"V-FUSE Training - Epoch {epoch}", unit="batches") as loader:
            for i, data in enumerate(loader):
                self.optimizer.zero_grad()
                to_device(data, self.device)

                # run model forward pass
                outputs = self.model(data)
                
                # compute loss and advance optimizer
                loss = compute_loss(data, outputs, self.cfg)
                loss["loss"].backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.cfg["training"]["gradient_clip"])
                self.optimizer.step()

                # clean up gpu mem
                torch.cuda.empty_cache()

                loss_sum["loss"] += float(loss["loss"].detach().item())
                loss_sum["depth"] += float(loss["depth"].detach().item())
                loss_sum["coverage"] += float(loss["coverage"].detach().item())
                loss_sum["radius"] += float(loss["radius"].detach().item())
                loss_sum["one_acc"] += float(loss["one_acc"].detach().item())
                loss_sum["three_acc"] += float(loss["three_acc"].detach().item())

                loader.set_postfix( \
                        total=f"{loss_sum['loss']/(i+1):4.2f}", \
                        depth=f"{loss_sum['depth']/(i+1):4.2f}", \
                        coverage=f"{loss_sum['coverage']/(i+1):4.2f}", \
                        radius=f"{loss_sum['radius']/(i+1):4.2f}", \
                        less1_acc=f"{loss_sum['one_acc']*100/(i+1):3.2f}%", \
                        less3_acc=f"{loss_sum['three_acc']*100/(i+1):3.2f}%")
        return

    def val_step(self, epoch):
        self.model.eval()
        loss_sum = {
                "loss": 0.0,
                "depth": 0.0,
                "coverage": 0.0,
                "radius": 0.0,
                "one_acc": 0.0,
                "three_acc": 0.0
                }

        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"V-FUSE Validation - Epoch {epoch}", unit="batches") as loader:
                for i, data in enumerate(loader):
                    to_device(data, self.device)

                    # run model forward pass
                    outputs = self.model(data)
                    loss = compute_loss(data, outputs, self.cfg)
                    torch.cuda.empty_cache()

                    loss_sum["loss"] += float(loss["loss"].detach().item())
                    loss_sum["depth"] += float(loss["depth"].detach().item())
                    loss_sum["coverage"] += float(loss["coverage"].detach().item())
                    loss_sum["radius"] += float(loss["radius"].detach().item())
                    loss_sum["one_acc"] += float(loss["one_acc"].detach().item())
                    loss_sum["three_acc"] += float(loss["three_acc"].detach().item())
                    loader.set_postfix( \
                            total=f"{loss_sum['loss']/(i+1):4.2f}", \
                            depth=f"{loss_sum['depth']/(i+1):4.2f}", \
                            coverage=f"{loss_sum['coverage']/(i+1):4.2f}", \
                            radius=f"{loss_sum['radius']/(i+1):4.2f}", \
                            less1_acc=f"{loss_sum['one_acc']*100/(i+1):3.2f}%", \
                            less3_acc=f"{loss_sum['three_acc']*100/(i+1):3.2f}%")
            return

    def inference(self):
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.data_loader, desc=f"V-FUSE Inference", unit="batches") as loader:
                for i, data in enumerate(loader):
                    to_device(data, self.device)
                    ref_index = int(data["ref_index"])
                    near_depth = data["cams"][0,0,1,3,0].item()
                    far_depth = data["cams"][0,0,1,3,3].item()

                    outputs = self.model(data)
                    torch.cuda.empty_cache()

                    # save fused depth
                    fused_depth_file = os.path.join(self.fused_depth_path, f"{ref_index:08d}.pfm")
                    fused_depth = outputs["fused_depth"][0,0].cpu().numpy()
                    write_pfm(fused_depth_file, fused_depth)
                    display_map(fused_depth_file[:-4]+"_disp.png", fused_depth, far_depth, near_depth)

                    # save fused confidence
                    fused_conf_file = os.path.join(self.fused_conf_path, f"{ref_index:08d}.pfm")
                    fused_conf = outputs["fused_conf"][0,0].cpu().numpy()
                    write_pfm(fused_conf_file, fused_conf)
                    display_map(fused_conf_file[:-4]+"_disp.png", fused_conf, 1, 0)

                    # save input depth
                    input_depth_file = os.path.join(self.input_depth_path, f"{ref_index:08d}.pfm")
                    input_depth = data["depths"][0,0,0].cpu().numpy()
                    write_pfm(input_depth_file, input_depth)
                    display_map(input_depth_file[:-4]+"_disp.png", input_depth, far_depth, near_depth)

                    # save input confidence
                    input_conf_file = os.path.join(self.input_conf_path, f"{ref_index:08d}.pfm")
                    input_confs = data["confs"][0,0,0].cpu().numpy()
                    write_pfm(input_conf_file, input_confs)
                    display_map(input_conf_file[:-4]+"_disp.png", input_confs, 1, 0)

                    # save camera
                    cam_file = os.path.join(self.cam_path, f"{ref_index:08d}.txt")
                    cam = data["cams"][0,0].cpu().numpy()
                    write_cam_sfm(cam_file, cam)
        return
