import os
import json
import hashlib
import numpy as np
from pathlib import Path
import torch
from torch import nn
import numpy as np
from thop import profile
from PIL import Image
import shutil

import kornia
import torchvision
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

##################
from models.encoder_decoder import FED
from models.network_scunet import SCUNet
##################

def load(model, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)

def save_image(
    tensor,
    fp,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format = None,
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, value_range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)

def save_images(sigma, folder, name, *args, **kwargs):
    imgs = []
    for i, img in enumerate(args):
        imgs.append((img.cpu() + 1) / 2)

    stacked_images = torch.cat(imgs, dim=0)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, 'sigma-{}--{}.png'.format(sigma, name))
    save_image(stacked_images, filename, 31, normalize=False)


def generate_binary_seed(seed_str: str) -> int:
    seed_str = seed_str.lower().replace("\\", "/").split('/')[-1]
    seed_str = seed_str.split('.')[0]
    hash_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
    return int.from_bytes(hash_bytes[:4], byteorder="big")


def generate_binary_data(seed: int, length: int = 30) -> list:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=length).tolist()
## new ##

def psnr_ssim_acc(image, H_img):
    # psnr
    H_psnr = kornia.metrics.psnr(
        ((image + 1) / 2).clamp(0, 1),
        ((H_img.detach() + 1) / 2).clamp(0, 1),
        1,
    )
    return H_psnr

def get_class_batch(input_root):
    class_imgs = []
    class_message = []
    classes = [d.name for d in os.scandir(input_root) if d.is_dir()]
    classes.sort()

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    for name in classes:
        cl = os.path.join(input_root, name)
        files = os.listdir(cl)
        files.sort()
        img_path = os.path.join(cl, files[0])
        image = Image.open(img_path).convert('RGB')
        img = img_transform(image)
        seed = generate_binary_seed(img_path)
        binary_data = generate_binary_data(seed, 30)
        class_imgs.append(img)
        class_message.append(torch.tensor(binary_data, dtype=torch.float32))

    batch_class_img = torch.stack(class_imgs, dim=0)
    batch_class_message = torch.stack(class_message, dim=0)
    batch_class_img = batch_class_img.to(device)
    batch_class_message = batch_class_message.to(device)
    return batch_class_img, batch_class_message


def calculate_Metric(name, n, input_root):
    dataset = ImageProcessingDataset(input_root)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    bitwise_avg_err_n_history = []
    bitwise_avg_err_r_history = []
    bitwise_avg_err_g_history = []
    SCUNet_H_psnrs = []
    GaussianBlur_H_psnrs = []
    L_psnrs = []
    N_psnrs = []
    diff_w_r_mses = []
    diff_w_g_mses = []
    with torch.no_grad():
        for data in dataloader:
            inputs, indices, message = data
            noise = torch.Tensor(np.random.normal(0, n, inputs.shape)/128.).to(device)
            inputs = inputs.to(device)

            message = message.to(device)
            #####################
            output_img, left_noise = fed([inputs, message])
            output_img_n = output_img + noise
            output_img_g = gaussian_blur(output_img_n)
            output_img_r = scunet(output_img_n)

            guass_noise = torch.zeros(left_noise.shape).to(device)

            ## new ##
            diff_w = output_img - inputs
            diff_r = output_img_r - inputs
            diff_g = output_img_g - inputs

            diff_w_r_mse = mse_loss(diff_w, diff_r)
            diff_w_g_mse = mse_loss(diff_w, diff_g)
            ## new ##

            _, decoded_messages_n = fed([output_img_n,guass_noise], rev=True)
            _, decoded_messages_r = fed([output_img_r,guass_noise], rev=True)
            _, decoded_messages_g = fed([output_img_g,guass_noise], rev=True)
            ####################                
            decoded_rounded_n = decoded_messages_n.detach().cpu().numpy().round().clip(0, 1)
            bitwise_avg_err_n = np.sum(np.abs(decoded_rounded_n - message.detach().cpu().numpy())) / (
                    batch_size * 30)

            decoded_rounded_r = decoded_messages_r.detach().cpu().numpy().round().clip(0, 1)
            bitwise_avg_err_r = np.sum(np.abs(decoded_rounded_r - message.detach().cpu().numpy())) / (
                    batch_size * 30)

            decoded_rounded_g = decoded_messages_g.detach().cpu().numpy().round().clip(0, 1)
            bitwise_avg_err_g = np.sum(np.abs(decoded_rounded_g - message.detach().cpu().numpy())) / (
                    batch_size * 30)
            SCUNet_H_psnr = psnr_ssim_acc(output_img.cpu(), output_img_r.cpu())
            GaussianBlur_H_psnr = psnr_ssim_acc(output_img.cpu(), output_img_g.cpu())
            L_psnr = psnr_ssim_acc(output_img.cpu(), output_img_n.cpu())
            N_psnr = psnr_ssim_acc(inputs.cpu(), (noise + inputs).cpu())

            SCUNet_H_psnrs.append(SCUNet_H_psnr)
            GaussianBlur_H_psnrs.append(GaussianBlur_H_psnr)

            L_psnrs.append(L_psnr)
            N_psnrs.append(N_psnr)
            bitwise_avg_err_n_history.append(bitwise_avg_err_n)
            bitwise_avg_err_r_history.append(bitwise_avg_err_r)
            bitwise_avg_err_g_history.append(bitwise_avg_err_g)
            ## new ##
            diff_w_r_mses.append(diff_w_r_mse.cpu())
            diff_w_g_mses.append(diff_w_g_mse.cpu())

    clean = 1 - np.mean(bitwise_avg_err_n_history)
    noise = 1 - np.mean(bitwise_avg_err_n_history)
    SCUNet_recover = 1 - np.mean(bitwise_avg_err_r_history)
    GaussianBlur_recover = 1 - np.mean(bitwise_avg_err_g_history)
    SCUNet_revover_rate = (SCUNet_recover - noise) / (clean - noise+1e-6)
    GaussianBlur_recover_rate = (GaussianBlur_recover - noise) / (clean - noise+1e-6)
    diff_w_r_mse_mean = np.mean(diff_w_r_mses)
    diff_w_g_mse_mean = np.mean(diff_w_g_mses)
    GaussianBlur_H_psnr = np.mean(GaussianBlur_H_psnrs)
    SCUNet_H_psnr = np.mean(SCUNet_H_psnrs)
    L_psnr = np.mean(L_psnrs)
    N_psnr = np.mean(N_psnrs)
    diff_w_g_mse = np.log10(diff_w_g_mse_mean)
    diff_w_r_mse = np.log10(diff_w_r_mse_mean)
                                       
    row = {'id': name}
    row['sigma'] = n
    row['clean_accuracy'] = clean * 100
    row['GaussianBlur_recovery_rate'] = GaussianBlur_recover_rate * 100
    row['SCUNet_recovery_rate'] = SCUNet_revover_rate * 100

    row['noise_image_accuracy'] = noise * 100
    row['GaussianBlur_accuracy'] = GaussianBlur_recover * 100
    row['SCUNet_accuracy'] = SCUNet_recover * 100

    row['GaussianBlur_psnr_wm_to_r'] = GaussianBlur_H_psnr
    row['SCUNet_psnr_wm_to_r'] = SCUNet_H_psnr
    row['L_psnr_wm_to_n'] = L_psnr
    row['N_psnr'] = N_psnr

    row['log10_diff_w_g_mse'] = diff_w_g_mse
    row['log10_diff_w_r_mse'] = diff_w_r_mse
    return row

def save_denoise_img(n):
    batch_class_img, batch_class_message = get_class_batch(input_root)
    with torch.no_grad():
        noise = torch.Tensor(np.random.normal(0, n, batch_class_img.shape)/128.).to(device)
        output_img, left_noise = fed([batch_class_img, batch_class_message])
        output_img_n = output_img + noise
        output_img_g = gaussian_blur(output_img_n)
        output_img_r = scunet(output_img_n)

        save_images(n, './img', 'Spatial', output_img, output_img_n, output_img_g, output_img_r)

        diff_w = output_img - batch_class_img
        diff_g = output_img_g - batch_class_img
        diff_r = output_img_r - batch_class_img

        o_dw = output_img.max() / diff_w.max()
        o_dg = output_img.max() / diff_g.max()
        o_dr = output_img.max() / diff_r.max()

        diff_w = diff_w*o_dw
        diff_g = diff_g*o_dg
        diff_r = diff_r*o_dr

        save_images(n, './img', 'Diff_I_D', diff_w, diff_g, diff_r)

        diff_g_w = diff_g - diff_w
        diff_r_w = diff_r - diff_w

        o_dg_w = output_img.max() / diff_g_w.max()
        o_dr_w = output_img.max() / diff_r_w.max()

        diff_g_w = diff_g_w*o_dg_w
        diff_r_w = diff_r_w*o_dr_w

        save_images(n, './img', 'Diff_M_D', diff_g_w, diff_r_w)

class ImageProcessingDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.image_paths = []
        self.rel_dirs = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        

        for root, _, files in os.walk(root_dir):
            rel_dir = os.path.relpath(root, root_dir)
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, f))
                    self.rel_dirs.append(rel_dir)

    def __len__(self):
        return len(self.image_paths)
    
    def generate_binary_seed(self, seed_str: str) -> int:
        seed_str = seed_str.lower().replace("\\", "/").split('/')[-1]
        seed_str = seed_str.split('.')[0]
        hash_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big")

    def generate_binary_data(self, seed: int, length: int = 30) -> list:
        rng = np.random.RandomState(seed)
        return rng.randint(0, 2, size=length).tolist()

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            seed = self.generate_binary_seed(img_path)
            binary_data = self.generate_binary_data(seed, 30)
            return self.transform(img), idx, torch.tensor(binary_data, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None, idx



if __name__ == "__main__":

    input_root = "../gtos128_all/val"
    batch_size = 32
    num_workers = 4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#######################################
    fed = FED(True, 30)
    fed.to(device)
    load(fed, './experiments/gtos_GN_25/fed_66_psnr-29.97329.pt')
#######################################

    scunet = SCUNet(in_nc=3,config=[4,4,4,4,4,4,4],dim=64)

    scunet.load_state_dict(torch.load('../SCUNet/runs/gtos_FIN_GN_29-2025-04-13-04:55-train/checkpoint/gtos_FIN_GN_29--epoch-4.pth')['network'], strict=True)
    scunet.to(device)
    scunet.eval()

    dataset = ImageProcessingDataset(input_root)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    gaussian_blur = transforms.GaussianBlur(
        kernel_size=7,
        sigma=1.0
    )
    class_imgs = []
    class_message = []
    classes = [d.name for d in os.scandir(input_root) if d.is_dir()]
    classes.sort()
    ## new ##
    mse_loss = torch.nn.MSELoss(reduction='mean')
    ## new ##
    results = []
    sigma = [0, 15, 25, 50, 75]
    for n in sigma:
        row = calculate_Metric('ALL', n, input_root)
        results.append(row)
        save_denoise_img(n)

    for name in classes:
        img_path_data = os.path.join(input_root, name)
        for n in sigma:
            row = calculate_Metric(name, n, img_path_data)
            results.append(row)
                        
    df = pd.DataFrame(results)
    df.to_csv('./result_FIN_GN_29.csv', index=False) # name
    print("finish")