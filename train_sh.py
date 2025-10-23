import os
import yaml
import argparse
from tqdm import tqdm
from ema_pytorch import EMA
import matplotlib.pyplot as plt
from setproctitle import setproctitle
import torch
import torch.distributed as dist
import warnings; warnings.filterwarnings("ignore")

from datasets import get_dataset
from model import DDPM, UNet_new
from utils import Config, get_optimizer, init_seeds, reduce_tensor, DataLoaderDDP, print0



def train(opt):
    """
    Train the model using the given configuration.
    
    Args:
        opt (Namespace): Training options.
    """
    yaml_path = f'config/{opt.system}.yaml'
    local_rank = opt.local_rank
    use_amp = opt.use_amp
    multi_gpu = opt.multi_gpu


    # Load configuration
    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    opt = Config(opt)
    
    # Device setup
    device = f"cuda:{local_rank}" if multi_gpu else "cuda:2"
    diff = DDPM(
        nn_model=UNet_new(**opt.network),
        **opt.diffusion,
        device=device,
    )
    opt.save_dir += f'/{opt.dataset}/'
    diff.to(device)
    
    params = sum(p.numel() for p in diff.parameters() if p.requires_grad) / 1e6
    print0(f"nn model # params: {params:.1f}M")
    # Directories for model checkpoints and visualization
    model_dir = os.path.join(opt.save_dir, "ckpts_55")
    vis_dir = os.path.join(opt.save_dir, "visual_55")   
    if not multi_gpu or local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    # EMA setup (only on rank 0 or single GPU)
    if not multi_gpu or local_rank == 0:
        ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1)
        ema.to(device)

    # Convert to SyncBatchNorm and DistributedDataParallel
    if multi_gpu:
        diff = torch.nn.SyncBatchNorm.convert_sync_batchnorm(diff)
        diff = torch.nn.parallel.DistributedDataParallel(
            diff, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )

    # Dataset and dataloader
    train_set = get_dataset(name=opt.dataset)
    print0(f"Train dataset size: {len(train_set)}")

    if multi_gpu:
        train_loader, sampler = DataLoaderDDP(
            train_set,
            batch_size=opt.batch_size,
            shuffle=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4
        )
        sampler = None

    # Optimizer and scaler
    lr = opt.lrate * (dist.get_world_size() if multi_gpu else 1)
    print0(f"Learning rate = {opt.lrate:.4e} * {dist.get_world_size() if multi_gpu else 1}")
    optim = get_optimizer(diff.parameters(), opt, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load checkpoint if specified
    if opt.load_epoch != -1:
        target = os.path.join(model_dir, f"model_{opt.load_epoch}.pth")
        print0(f"Loading model from {target}")
        checkpoint = torch.load(target, map_location=device)
        diff.load_state_dict(checkpoint['MODEL'], strict=False)
        if not multi_gpu or local_rank == 0:
            ema.load_state_dict(checkpoint['EMA'], strict=False)
        optim.load_state_dict(checkpoint['opt'])

    # Training loop
    for ep in range(opt.load_epoch + 1, opt.n_epoch):# Validation and checkpoint saving
        # Warm-up learning rate adjustment
        for g in optim.param_groups:
            g['lr'] = lr * min((ep + 1.0) / opt.warm_epoch, 1.0)

        if multi_gpu and sampler is not None:
            sampler.set_epoch(ep)
            dist.barrier()

        # Training epoch
        diff.train()
        if not multi_gpu or local_rank == 0:
            current_lr = optim.param_groups[0]['lr']
            print(f"Epoch {ep}, Learning Rate: {current_lr:.4e}")
            loss_ema = None
            epoch_loss = 0.0  
            num_batches = 0 
            pbar = tqdm(train_loader, desc="Training")
        else:
            pbar = train_loader

        for x in pbar:
            # x: B, C, H, W
            optim.zero_grad()
            x= x.to(device)

            loss = diff(x, use_amp=use_amp)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(diff.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            # Logging
            if multi_gpu:
                dist.barrier()
                loss = reduce_tensor(loss)
            if not multi_gpu or local_rank == 0:
                ema.update()
                loss_ema = loss.item() if loss_ema is None else 0.95 * loss_ema + 0.05 * loss.item()
                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_description(f"Loss: {loss_ema:.4e}")  
        if not multi_gpu or local_rank == 0:
            avg_loss = epoch_loss / num_batches 
            print(f"Epoch {ep} - Average Loss: {avg_loss:.4e}")

        # Validation and checkpoint saving     
        if not multi_gpu or local_rank == 0:
            if ep % 100 == 0 or ep == opt.n_epoch - 1:
                ema_sample_method = ema.ema_model.ddim_sample
                # ema_sample_method = ema.ema_model.sample
                ema.ema_model.eval()
                with torch.no_grad():
                    idx = torch.randint(0, x.size(0), (6,))
                    x_real = x[idx] # shape: (6, C, H, W)
                    n_sample = x_real.shape[0]
                    size = x_real.shape[1:]
                    x_gen = ema_sample_method(n_sample, size) # shape: (6, C, H, W)
                
                nmse = torch.mean((x_gen - x_real).pow(2)).item() / torch.mean(x_real.pow(2)).item()
                print(f"Val NMSE: {nmse:.4f}")
                
                width = 12 if opt.dataset != 'cy' else 24
                plt.figure(figsize=(width, 4))
                for idx in range(6):
                    plt.subplot(2, 6, idx + 1)
                    plt.imshow(x_real[idx, 0].T.cpu().numpy(), cmap='coolwarm')
                    plt.title(f"Frame {idx} (Real)")
                    plt.axis('off')
                    plt.subplot(2, 6, idx + 7)
                    plt.imshow(x_gen[idx, 0].T.cpu().numpy(), cmap='coolwarm')
                    plt.title(f"Frame {idx} (Gen)")
                    plt.axis('off')
                plt.savefig(f"{vis_dir}/val_{ep}.png", bbox_inches='tight', dpi=300)
                
            if ep % 20 == 0 or ep == opt.n_epoch - 1:   
                if opt.save_model:
                    checkpoint = {
                        'MODEL': diff.state_dict(),
                        'EMA': ema.state_dict(),
                        'opt': optim.state_dict(),
                    }
                    save_path = os.path.join(model_dir, f"model_{ep}.pth")
                    torch.save(checkpoint, save_path)
                    print(f"Saved model at {save_path}")


if __name__ == "__main__":
    # export OMP_NUM_THREADS=4
    # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    # python -m torch.distributed.run --master_port=25640 --nproc_per_node=2 train_ns.py --use_amp --multi_gpu --system sh
    
    
    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument("--system", type=str, default="sh", help="Path to config YAML file.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Node rank for distributed training.")
    parser.add_argument("--use_amp", action='store_true', default=True, help="Enable automatic mixed precision.")
    parser.add_argument("--multi_gpu", action="store_true", default=True, help="Enable multi-GPU training.")
    opt = parser.parse_args()
    
    # Initialize for multi-GPU or single-GPU
    if opt.multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "3,6"
        init_seeds(no=opt.local_rank)
        from datetime import timedelta
        opt.local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=30),
        )
        torch.cuda.set_device(opt.local_rank)
        
    else:
        init_seeds()

    train(opt)
