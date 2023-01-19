import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import numpy as np
import wandb
import os
import shutil
import open_clip

import webdataset as wds
from webdataset.handlers import warn_and_continue

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from torchtools.utils import Diffuzz
from modules import VQModel, to_latent, from_latent
from modules_alt import DiffusionModel
from utils import WebdatasetFilter

# PARAMETERS
updates = 1500000 # 500000
warmup_updates = 10000
batch_size = 2048
grad_accum_steps = 1
max_iters = updates * grad_accum_steps
print_every = 500 * grad_accum_steps
lr = 3e-5 # 1e-4 # 3e-4
scaler_min_scale = 128

# dataset_path = "pipe:aws s3 cp s3://s-laion/improved-aesthetics-laion-2B-en-subsets/aesthetics_tars/{000000..060207}.tar -"
# dataset_path = "pipe:aws s3 cp s3://deep-floyd-s3/datasets/{laion_cleaned-part1/{00000..79752}.tar,laion_cleaned-part2/{00000..94330}.tar,laion_cleaned-part3/{00000..94336}.tar,laion_cleaned-part4/{00000..94340}.tar,laion_cleaned-part5/{00000..94333}.tar,laion_cleaned-part6/{00000..77178}.tar} -"
# dataset_path = "pipe:aws s3 cp s3://s-datasets/laion-high-resolution/{00000..17535}.tar -"
dataset_path = "pipe:aws s3 cp s3://stability-aws/laion-a-native/{part-0/{00000..18699}.tar,part-1/{00000..18699}.tar,part-2/{00000..18699}.tar,part-3/{00000..18699}.tar,part-4/{00000..18699}.tar} -"
clip_model_name = ('ViT-H-14', 'laion2b_s32b_b79k')
output_path = "../../output/arroz_con_cosas_1b_attn/"
checkpoint_path = "../../models/arroz_con_cosas/clip2img_v5_1b_attn.pt"

wandv_project = "ArrozConCosas"
wandv_entity = "babbleberns"
# wandb_run_name = "clip2img_v5_1b_attn"
wandb_run_name = "clip2img_v5_1b_attn_stage2"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(640), # 512 + 128
    torchvision.transforms.RandomCrop(640),
])
clip_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    )
])
def identity(x):
    return x

# crop = torchvision.transforms.RandomResizedCrop(256, scale=(0.4, 0.6), ratio=(1.0, 1.0))
crop = torchvision.transforms.RandomResizedCrop(512, scale=(0.8, 1.0), ratio=(1.0, 1.0))

def ddp_setup(rank, world_size, n_node, node_id): # <--- DDP
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "33751"
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        rank=rank+node_id*world_size, world_size=world_size*n_node,
        init_method="file:///fsx/home-pablo/src/arroz_con_cosas/dist_file_v5_1b_attn",
    )
    print(f"[GPU {rank+node_id*world_size}] READY")

def train(gpu_id, world_size, n_nodes):
    node_id = int(os.environ["SLURM_PROCID"])
    ddp_setup(gpu_id, world_size, n_nodes, node_id) # <--- DDP
    device = torch.device(gpu_id)
    
    # --- PREPARE DATASET ---
    # PREPARE DATASET
    dataset = wds.WebDataset(
        dataset_path, resampled=True, handler=warn_and_continue
    ).select(
        WebdatasetFilter(min_size=512, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99)
    ).shuffle(690, handler=warn_and_continue).decode(
        "pilrgb", handler=warn_and_continue
    ).to_tuple(
        "jpg", "txt", handler=warn_and_continue
    ).map_tuple(
        transforms, identity, handler=warn_and_continue
    )
    real_batch_size = batch_size//(world_size*n_nodes*grad_accum_steps)
    dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True)

    if gpu_id == 0 and node_id == 0:
        print("REAL BATCH SIZE / DEVICE:", real_batch_size)

    # --- PREPARE MODELS ---
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device) if os.path.exists(checkpoint_path) else None
    except RuntimeError as e:
        if os.path.exists(f"{checkpoint_path}.bak"):
            os.remove(checkpoint_path)
            shutil.copyfile(f"{checkpoint_path}.bak", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            raise e

    # - utils - 
    diffuzz = Diffuzz(device=device)
    # diffuzz = Diffuzz(device=device, cache_steps=10000)

    # - vqmodel -
    vqmodel = VQModel().to(device)
    vqmodel.load_state_dict(torch.load("../../models/arroz_con_cosas/vqwatercolor_v1.pt", map_location=device))
    vqmodel.eval().requires_grad_(False)

    # - class conditional embedding -
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name[0], pretrained=clip_model_name[1], cache_dir="/fsx/home-pablo/.cache", device=device)
    clip_model.eval().requires_grad_(False)

    # - denoisegic - 
    model = DiffusionModel().to(device)
    
    # load checkpoints & prepare ddp
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])

    model = DDP(model, device_ids=[gpu_id], output_device=device) # <--- DDP

    if gpu_id == 0 and node_id == 0: # <--- DDP
        print("Num trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # - SETUP WANDB - 
    if gpu_id == 0 and node_id == 0: # <--- DDP
        run_id = checkpoint['wandb_run_id'] if checkpoint is not None else wandb.util.generate_id()
        wandb.init(project=wandv_project, name=wandb_run_name, entity=wandv_entity, id=run_id, resume="allow")
        # wandb.watch(model)

    # SETUP OPTIMIZER, SCHEDULER & CRITERION
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_updates)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.last_epoch = checkpoint['scheduler_last_step']
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(growth_interval=100)
    if checkpoint is not None and 'grad_scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

    start_iter = 1
    grad_norm = torch.tensor(0, device=device)
    if checkpoint is not None:
        start_iter = checkpoint['iter'] + 1
        if gpu_id == 0 and node_id == 0: # <--- DDP
            print("RESUMING TRAINING FROM ITER ", start_iter)  

    ema_loss = None
    if checkpoint is not None:
        ema_loss = checkpoint['metrics']['ema_loss']

    if checkpoint is not None:
        del checkpoint # cleanup memory
        torch.cuda.empty_cache() 
    
    # -------------- START TRAINING --------------    
    dataloader_iterator = iter(dataloader)
    pbar = tqdm(range(start_iter, max_iters+1)) if (gpu_id == 0 and node_id == 0) else range(start_iter, max_iters+1) # <--- DDP
    model.train()
    for it in pbar:
        images, captions = next(dataloader_iterator)
        images = images.to(device)
        with torch.no_grad():
            if np.random.rand() < 0.05:
                image_embeddings = images.new_zeros(images.size(0), 1024)
            else:
                with torch.cuda.amp.autocast():
                    image_embeddings = clip_model.encode_image(clip_preprocess(images)).float()
                if np.random.rand() < 0.1: # 10% of the time add random gausian noise to the image embeddings (yeah, I know, a lot of augmentation, but I want to make the model even more robust!!!)
                    noise_scale = torch.rand(image_embeddings.size(0), device=device) * 0.5 # 0 to 50%
                    image_embeddings, _ = diffuzz.diffuse(image_embeddings, noise_scale)

            images = crop(images)
            t = 1-torch.rand(images.size(0), device=device)
            qe = to_latent(images, vqmodel)
            noised_xq, noise = diffuzz.diffuse(qe, t)

        with torch.cuda.amp.autocast():
            pred_noise = model(noised_xq, t, image_embeddings)
            loss = criterion(pred_noise, noise)
            loss_adjusted = loss / grad_accum_steps

        # loss_adjusted.backward()
        scaler.scale(loss_adjusted).backward()
        if it % grad_accum_steps == 0 or it == max_iters:
            # optimizer.step()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0) 
            scaler.step(optimizer)
            scaler.update()
            if scaler._scale < scaler_min_scale:
                scaler._scale = torch.tensor(scaler_min_scale).to(scaler._scale)
            scheduler.step()
            optimizer.zero_grad()

        ema_loss = loss.item() if ema_loss is None else ema_loss * 0.99 + loss.item() * 0.01

        if gpu_id == 0 and node_id == 0: # <--- DDP
            pbar.set_postfix({
                'bs': images.size(0),
                'loss': ema_loss, 
                'grad_norm': grad_norm.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'total_steps': scheduler.last_epoch,
            })
            wandb.log({
                'loss': ema_loss, 
                'grad_norm': grad_norm.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'total_steps': scheduler.last_epoch,
            })

        if gpu_id == 0 and node_id == 0 and (it == 1 or it % print_every == 0 or it == max_iters): # <--- DDP
            print(f"ITER {it}/{max_iters} - loss {ema_loss}")

            model.eval()
            images = next(dataloader_iterator)[0][:8].to(device)
            with torch.no_grad():
                image_embeddings = clip_model.encode_image(clip_preprocess(images)).float()
                images = crop(images)
                
                t = 1-torch.rand(images.size(0), device=device)
                qe = to_latent(images, vqmodel)
                noised_xq, noise = diffuzz.diffuse(qe, t)

                pred_noise = model(noised_xq, t, image_embeddings)
                pred = diffuzz.undiffuse(noised_xq, t, torch.zeros_like(t), pred_noise)
                sampled = diffuzz.sample(model, {'c': image_embeddings}, (image_embeddings.size(0), 4, images.size(-2)//8, images.size(-1)//8),)[-1]

                noised_images = from_latent(noised_xq, vqmodel).clamp(0, 1)
                pred_images = from_latent(pred, vqmodel).clamp(0, 1)
                sampled_images = from_latent(sampled, vqmodel).clamp(0, 1)
            model.train()

            torchvision.utils.save_image(torch.cat([
                torch.cat([i for i in images.cpu()], dim=-1),
                torch.cat([i for i in noised_images.cpu()], dim=-1),
                torch.cat([i for i in pred_images.cpu()], dim=-1),
                torch.cat([i for i in sampled_images.cpu()], dim=-1),
            ], dim=-2), f'{output_path}{it:06d}.png')

            try:
                os.remove(f"{checkpoint_path}.bak")
            except OSError:
                pass

            try:
                os.rename(checkpoint_path, f"{checkpoint_path}.bak")
            except OSError:
                pass
                
            torch.save({
                'state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_last_step': scheduler.last_epoch,
                'iter' : it,
                'metrics' : {
                    'ema_loss': ema_loss,
                },
                'grad_scaler_state_dict': scaler.state_dict(),
                'wandb_run_id': run_id,
            }, checkpoint_path)

            log_data = [ [wandb.Image(sampled_images[i])] + [wandb.Image(images[i])] for i in range(len(images))]
            log_table = wandb.Table(data=log_data, columns=["Sampled", "Orig"])
            wandb.log({"Log": log_table})

            del pred, sampled, noised_images, pred_images, sampled_images, log_data, log_table      
        del pred_noise, images, image_embeddings, qe, t, noised_xq, noise, loss, loss_adjusted
        torch.cuda.empty_cache()

    destroy_process_group() # <--- DDP

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    n_node = 8
    mp.spawn(train, args=(world_size, n_node), nprocs=world_size) # <--- DDP
    

