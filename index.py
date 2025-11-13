# index.py
import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn

# --- Config ---
LATENT_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/generator.pth"

# --- Generator (must match training) ---
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128*7*7),
            nn.BatchNorm1d(128*7*7),
            nn.ReLU(True),

            nn.Unflatten(1, (128, 7, 7)),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# --- Load Generator ---
@st.cache_resource
def load_generator():
    model = Generator(LATENT_DIM).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

generator = load_generator()

# --- UI ---
st.title("🖼️ GAN Art Generator (MNIST)")

num_images = st.slider("Number of images to generate", 1, 25, 9)
if st.button("Generate Images"):
    z = torch.randn(num_images, LATENT_DIM, device=DEVICE)
    gen_imgs = generator(z).detach().cpu()
    gen_imgs = (gen_imgs + 1) / 2  # Rescale from [-1,1] → [0,1]

    grid = make_grid(gen_imgs, nrow=int(num_images**0.5), normalize=True)

    plt.figure(figsize=(6,6))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    st.pyplot(plt)
