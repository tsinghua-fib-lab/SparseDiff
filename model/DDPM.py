from functools import partial
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F
import time
import numpy as np

def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5


def linear_beta_schedule(timesteps, beta1, beta2):
    assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    return torch.linspace(beta1, beta2, timesteps)


def schedules(betas, T, device, type='DDPM'):
    beta1, beta2 = betas
    schedule_fn = partial(linear_beta_schedule, beta1=beta1, beta2=beta2)  

    if type == 'DDPM':
        beta_t = torch.cat([torch.tensor([0.0]), schedule_fn(T)])
    elif type == 'DDIM':
        beta_t = schedule_fn(T + 1)
    else:
        raise NotImplementedError()
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    ma_over_sqrtmab = (1 - alpha_t) / sqrtmab

    dic = {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "ma_over_sqrtmab": ma_over_sqrtmab,
    }
    return {key: dic[key].to(device) for key in dic}


# def get_observation_loss(mask, square_sparse, x):
   
#     error = (square_sparse - x) ** 2  # shape: (B, C, H, W)   
#     masked_error = error * mask.unsqueeze(0)
#     valid_points_per_channel = torch.sum(mask, dim=(1, 2))  # shape: (C,)
#     valid_points = valid_points_per_channel.view(1, -1, 1, 1) + 1e-8  
#     observation_loss = masked_error / valid_points
#     return observation_loss

def get_observation_loss_full(reconstructed, x):   
    error = (reconstructed - x) ** 2  # shape: (B, C, H, W)
    observation_loss = error / (128*128)

    return observation_loss

def periodic_laplacian(u, dx):
    return (
        torch.roll(u, 1, dims=-2) + torch.roll(u, -1, dims=-2) +
        torch.roll(u, 1, dims=-1) + torch.roll(u, -1, dims=-1) - 4 * u
    ) / dx**2

def periodic_biharmonic(u, dx):
    return periodic_laplacian(periodic_laplacian(u, dx), dx)

def get_sh_pde_loss(x_seq, data_opt):
    """
    x_seq: (T, 1, H, W) - sequence of predicted frames
    returns: pde_loss (T, 1, H, W)
    """
    dx = data_opt["dx"]
    dt = data_opt["dt"]
    r = data_opt["r"]
    g = data_opt["g"]

    T, _, H, W = x_seq.shape
    device = x_seq.device

    # --- Compute ∂u/∂t using conv1d ---
    # Reshape (T, 1, H, W) → (H*W, 1, T)
    u_flat = x_seq.permute(2, 3, 1, 0).reshape(-1, 1, T)  # (H*W, 1, T)

    # Central difference kernel
    deriv_kernel = torch.tensor([[[1.0, 0.0, -1.0]]], device=device, dtype=torch.float32) / (2 * dt)
    u_t_flat = F.conv1d(u_flat, deriv_kernel, padding=1)  # (H*W, 1, T)

    # Reshape back → (T, 1, H, W)
    u_t = u_t_flat.reshape(H, W, 1, T).permute(3, 2, 0, 1)

    # --- Fix boundaries: forward/backward difference ---
    u_t[0]  = (x_seq[1] - x_seq[0]) / dt
    u_t[-1] = (x_seq[-1] - x_seq[-2]) / dt

    # --- Compute RHS for each frame ---
    u = x_seq[:, 0, :, :]                         # (T, H, W)
    lap = periodic_laplacian(u, dx)               # (T, H, W)
    bih = periodic_biharmonic(u, dx)              # (T, H, W)
    rhs = r * u - 2 * lap - bih + g * u**2 - u**3

    # --- Final loss ---
    loss = (u_t[:, 0] - rhs) ** 2  # shape: (T, H, W)
    return loss.unsqueeze(1)       # shape: (T, 1, H, W)

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        ''' DDPM proposed by "Denoising Diffusion Probabilistic Models", and \
            DDIM sampler proposed by "Denoising Diffusion Implicit Models".

            Args:
                nn_model: A network (e.g. UNet) which performs same-shape mapping.
                device: The CUDA device that tensors run on.
            Parameters:
                betas, n_T
        '''
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad) / 1e6

        self.device = device
        self.ddpm_sche = schedules(betas, n_T, device, 'DDPM')
        self.ddim_sche = schedules(betas, n_T, device, 'DDIM')
        self.n_T = n_T
        self.loss = nn.MSELoss()


    def perturb(self, x, t=None):
        ''' Add noise to a clean image (diffusion process).

            Args:
                x: The normalized image tensor.
                t: The specified timestep ranged in `[1, n_T]`. Type: int / torch.LongTensor / None. \
                    Random `t ~ U[1, n_T]` is taken if t is None.
            Returns:
                The perturbed image, the corresponding timestep, and the noise.
        '''
        if t is None:
            t = torch.randint(1, self.n_T + 1, (x.shape[0], )).to(self.device)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).to(self.device).repeat(x.shape[0])

        noise = torch.randn_like(x)
        sche = self.ddpm_sche
        x_noised = (sche["sqrtab"][t, None, None, None] * x +   # broadcast
                    sche["sqrtmab"][t, None, None, None] * noise)
        return x_noised, t, noise

    def forward(self, x, use_amp=False):
        ''' Training with simple noise prediction loss.

            Args:
                x: The clean image tensor ranged in `[0, 1]`.
            Returns:
                The simple MSE loss.
        '''
        x = normalize_to_neg_one_to_one(x)
        x_noised, t, noise = self.perturb(x, t=None)

        with autocast(enabled=use_amp):
            return self.loss(noise, self.nn_model(x_noised, t / self.n_T))

    def sample(self, n_sample, size, notqdm=False, use_amp=False):
       
        ''' Sampling with DDPM sampler. Actual NFE is `n_T`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        sche = self.ddpm_sche
        x_i = torch.randn(n_sample, *size).to(self.device)  

        for i in tqdm(range(self.n_T, 0, -1), disable=notqdm):
            t_is = torch.tensor([i / self.n_T]).to(self.device).repeat(n_sample)  

            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0  

            alpha = sche["alphabar_t"][i]  
            eps, _ = self.pred_eps_(x_i, t_is, alpha, use_amp)

            mean = sche["oneover_sqrta"][i] * (x_i - sche["ma_over_sqrtmab"][i] * eps)
            variance = sche["sqrt_beta_t"][i] # LET variance sigma_t = sqrt_beta_t
            x_i = mean + variance * z

        return unnormalize_to_zero_to_one(x_i)
    
    
    def ddim_guided_sample_full_sh(self, n_sample, size, steps=100, eta=0.0, 
                        zeta_obs=0.0, zeta_pde=0.0, ratio = 0.1, 
                        reconstructed=None, data_opt=None, notqdm=False):
        '''
        Guided sampling with DDIM.

        Args:
            n_sample: batch size
            size: (C, H, W)
            steps: total number of DDIM steps (<< self.n_T)
            eta: noise scale (eta=0.0 is deterministic DDIM)
            zeta_obs: weight of observation gradient
            zeta_pde: weight of PDE gradient
            reconstructed: observation image for loss guidance (must be normalized to [0, 1])
            data_opt: additional data for PDE loss
        '''
        sche = self.ddim_sche
        x_i = torch.randn(n_sample, *size).to(self.device)

        if reconstructed is not None:
            reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)

        # Subsample time steps
        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            x_i = x_i.detach().clone()
            x_i.requires_grad_(True)

            t_is = torch.tensor([time / self.n_T], device=self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if (eta > 0 and time_next > 0) else 0

            # Predict x0 and eps
            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha)

            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()

            x_next = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

            # === Gradient guidance ===
            if reconstructed is not None:
                obs_loss = get_observation_loss_full(reconstructed, x_i)
                grad_obs = torch.autograd.grad(obs_loss.sum(), x_i, retain_graph=True)[0]

                if time > 0.2 * self.n_T:
                    x_next = x_next - zeta_obs * grad_obs
                else:
                    pde_loss = get_sh_pde_loss(x_i, data_opt) / (128 * 128)
                    grad_pde = torch.autograd.grad(pde_loss.sum(), x_i, retain_graph=True)[0]
                    x_next = x_next - 0.1 * (zeta_obs * grad_obs) - zeta_pde * grad_pde

            x_i = x_next

        return unnormalize_to_zero_to_one(x_i)
    
    
    def ddim_sample_from_reconstructed_sh(self, n_sample, size, steps=100, eta=0.0, 
                        zeta_obs=0.0, zeta_pde=0.0, ratio = 0.1, 
                        reconstructed=None, data_opt=None, notqdm=False):
        '''
        Guided sampling with DDIM.

        Args:
            n_sample: batch size
            size: (C, H, W)
            steps: total number of DDIM steps (<< self.n_T)
            eta: noise scale (eta=0.0 is deterministic DDIM)
            zeta_obs: weight of observation gradient
            zeta_pde: weight of PDE gradient
            reconstructed: observation image for loss guidance (must be normalized to [0, 1])
            data_opt: additional data for PDE loss
        '''
        sche = self.ddim_sche
        

        if reconstructed is not None:
            reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)
        x_i = reconstructed

        times = torch.arange(0, self.n_T/5, self.n_T/5 // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # print("denoising time pairs:", time_pairs)

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            x_i = x_i.detach().clone()
            x_i.requires_grad_(True)

            t_is = torch.tensor([time / self.n_T], device=self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if (eta > 0 and time_next > 0) else 0

            # Predict x0 and eps
            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha)

            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()

            x_next = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

            if reconstructed is not None:
                obs_loss = get_observation_loss_full(reconstructed, x_i)
                grad_obs = torch.autograd.grad(obs_loss.sum(), x_i, retain_graph=True)[0]

                if time > 0.1 * self.n_T:
                    x_next = x_next - zeta_obs * grad_obs
                else:
                    pde_loss = get_sh_pde_loss(x_i, data_opt) / (128 * 128)
                    grad_pde = torch.autograd.grad(pde_loss.sum(), x_i, retain_graph=True)[0]
                    x_next = x_next - 0.1 * (zeta_obs * grad_obs) - zeta_pde * grad_pde

            x_i = x_next

        return unnormalize_to_zero_to_one(x_i)
    
    def ddim_sample(self, n_sample, size, steps=100, eta=0.0, notqdm=False, use_amp=False):
        ''' Sampling with DDIM sampler. Actual NFE is `steps`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
                steps: The number of total timesteps.
                eta: controls stochasticity. Set `eta=0` for deterministic sampling.
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        sche = self.ddim_sche
        x_i = torch.randn(n_sample, *size).to(self.device)

        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # e.g. [(801, 601), (601, 401), (401, 201), (201, 1), (1, 0)]

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            t_is = torch.tensor([time / self.n_T]).to(self.device).repeat(n_sample)

            z = torch.randn(n_sample, *size).to(self.device) if time_next > 0 else 0

            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha, use_amp)
            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()
            x_i = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

        return unnormalize_to_zero_to_one(x_i)

    def pred_eps_(self, x, t, alpha, clip_x=True):
        def pred_eps_from_x0(x0):
            return (x - x0 * alpha.sqrt()) / (1 - alpha).sqrt()

        def pred_x0_from_eps(eps):
            return (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()

        # get prediction of x0
    
        with torch.no_grad():
            eps = self.nn_model(x, t).float()    
         
        denoised = pred_x0_from_eps(eps)

        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
            eps = pred_eps_from_x0(denoised)
        return eps, denoised
    
   