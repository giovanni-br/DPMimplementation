########################################
# main_experiment.py
########################################

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from pytorch_fid import fid_score

# 1) Import the Hugging Face DDPM pipeline (for the pretrained unet).
from diffusers import DDPMPipeline

# 2) Import your DPM-Solver code from dpmsolver_lib
from dpmsolver_lib import NoiseScheduleVP, model_wrapper, DPM_Solver

def main():
    # ------------------------------
    # A. Load the pretrained pipeline
    # ------------------------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    unet = pipeline.unet  # We'll call unet(...) in a custom function

    # Extract discrete-time betas from the pipeline’s scheduler
    betas = pipeline.scheduler.betas.to(device)

    # We create a NoiseScheduleVP in "discrete" mode with these betas.
    noise_schedule = NoiseScheduleVP(
        schedule='discrete',
        betas=betas,
        dtype=torch.float32
    )

    # -----------------------------------------------
    # B. Build a small wrapper around the HF unet model
    # -----------------------------------------------
    def hf_cifar10_unet_wrapper(x, t_input, **model_kwargs):
        # If t_input is a float or shape [batch], get a Python float or array
        if isinstance(t_input, float) or t_input.shape == ():  
            t_input = t_input.unsqueeze(0)
        # Map continuous [1/1000, 1] => discrete [0..999].
        t_input_discrete = torch.round((1.0 - t_input) * 999).long().to(device=x.device)
        t_input_discrete = torch.clamp(t_input_discrete, 0, 999)
        model_out = unet(sample=x, timestep=t_input_discrete, return_dict=False)[0]
        return model_out

    # -------------------------------------------------------
    # C. Create the 'model_fn' with your code’s model_wrapper
    # -------------------------------------------------------
    model_fn = model_wrapper(
        model=hf_cifar10_unet_wrapper,
        noise_schedule=noise_schedule,
        model_type="noise",      # Our UNet predicts noise (DDPM standard)
        guidance_type="uncond",  # No classifier guidance
        guidance_scale=1.0,
    )

    # ----------------------------------------------
    # D. Build the DPM_Solver object
    # ----------------------------------------------
    dpm_solver = DPM_Solver(
        model_fn=model_fn,
        noise_schedule=noise_schedule,
        algorithm_type="dpmsolver++"  # or "dpmsolver"
    )

    # A small helper to sample from the model using DPM-Solver:
    def sample_cifar10_dpm(solver, batch_size=1, steps=50, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        init_x = torch.randn(batch_size, 3, 32, 32, generator=generator, device=device)

        x_0 = solver.sample(
            x=init_x,
            steps=steps,
            t_start=1.0,
            t_end=1.0 / noise_schedule.total_N,  # e.g. 1/1000
            order=2,               # or 3
            skip_type="time_uniform",
            method="multistep",
            denoise_to_zero=False,
        )
        return x_0

    # --------------
    # E. FID vs. NFE
    # --------------
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("real_images", exist_ok=True)

    # 1) Save a small batch of real CIFAR-10 images for comparison
    cifar10_data = CIFAR10(root="./data", train=True, download=True, transform=ToTensor())
    for idx in range(10):
        real_img = cifar10_data[idx][0]
        save_image(real_img, f"real_images/{idx:04d}.png")

    # 2) Evaluate FID at different NFEs
    nfe_values = [10, 100]
    fid_scores = []

    for nfe in nfe_values:
        # Generate images
        for idx in range(10):
            with torch.no_grad():
                x_0 = sample_cifar10_dpm(dpm_solver, batch_size=1, steps=nfe, seed=idx)
            x_0_clamped = (x_0.clamp(-1,1) + 1)/2
            save_image(x_0_clamped, f"generated_images/{idx:04d}.png")

        # Compute FID
        fid_val = fid_score.calculate_fid_given_paths(
            ["generated_images", "real_images"],
            batch_size=10,
            device=device,
            dims=2048,
            num_workers=0  # Avoids multiprocessing spawn issues on Windows
        )
        fid_scores.append(fid_val)
        print(f"NFE = {nfe}, FID = {fid_val:.4f}")

    # 3) Plot FID vs. NFE
    plt.figure(figsize=(8,6))
    plt.plot(nfe_values, fid_scores, marker="x", label="DPM-Solver++")
    plt.xlabel("NFE")
    plt.ylabel("FID")
    plt.title("FID vs NFE on CIFAR-10 (DPM-Solver++)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------
# Windows requires this guard
# ---------------------------
if __name__ == "__main__":
    main()
