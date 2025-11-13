import streamlit as st
import torch
import torchvision.utils as vutils
from src.datascience.components.mode_trainer import Generator
import yaml
import os
import numpy as np

# Load config
with open("config/config.yaml") as f:
	config = yaml.safe_load(f)

LATENT_DIM = config["model_trainer"]["latent_dim"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Find generator checkpoint
possible_paths = [
	"models/generator_final.pth",
	"models/generator.pth",
	"artifacts/model_trainer/generator.pth"
]
ckpt_path = next((p for p in possible_paths if os.path.exists(p)), None)
if ckpt_path is None:
	st.error(f"No generator checkpoint found. Expected one of: {possible_paths}")
	st.stop()

# Load Generator (MNIST-style)
generator = Generator(LATENT_DIM).to(DEVICE)
try:
	generator.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
except Exception as e:
	st.error(f"Error loading checkpoint: {e}")
	st.stop()
generator.eval()

# UI
st.title("🎨 GAN Art Generator (MNIST mode)")

data_root = os.path.abspath("data")
st.info(f"MNIST dataset root: {data_root}")

num_images = st.slider("Number of images", 1, 16, 4)
auto_generate = st.checkbox("Auto generate on slider change", value=False)

def generate_and_display(n):
	# Sample noise for MNIST generator: shape (batch, latent_dim)
	z = torch.randn(n, LATENT_DIM, device=DEVICE)
	with torch.no_grad():
		gen_imgs = generator(z).cpu()         # expected shape: (N,1,H,W) or (N,C,H,W)

	# Denormalize from [-1,1] -> [0,1]
	gen_imgs = (gen_imgs + 1.0) / 2.0
	gen_imgs = torch.clamp(gen_imgs, 0.0, 1.0)

	# Make grid (CxHxW)
	grid = vutils.make_grid(gen_imgs, normalize=False, nrow=min(4, n))
	grid_np = grid.permute(1, 2, 0).numpy()  # H x W x C

	# If single channel, convert to 3-channel RGB for consistent display
	if grid_np.shape[2] == 1:
		grid_np = np.repeat(grid_np, 3, axis=2)

	st.image(grid_np, use_container_width=True)

# Generate immediately if auto_generate checked, otherwise on button press
if auto_generate:
	generate_and_display(num_images)
else:
	if st.button("Generate"):
		generate_and_display(num_images)

# Show last saved sample if available
final_img_path = "images/final_generated.png"
if os.path.exists(final_img_path):
	st.subheader("Last saved generation")
	st.image(final_img_path, use_container_width=True)
else:
	st.info("No saved generated image found at images/final_generated.png")
