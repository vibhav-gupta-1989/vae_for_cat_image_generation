# Variational Autoencoder (VAE) for Generating Cat Doodles 🐱

A compact PyTorch implementation of a **Variational Autoencoder** trained on the [Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) cat category. The model learns to compress 28×28 hand-drawn cats into a small latent code, reconstruct them, and generate brand-new cat drawings by sampling from that latent space.

The whole project lives in a single notebook: [`vae.ipynb`](vae.ipynb).

---

## The Intuition

Think of a VAE as a **very strict note-taker and a very good sketch artist working together**.

- The **encoder** looks at a cat drawing and writes a 32-number "description" of it on a sticky note.
- But the notes are deliberately *fuzzy*: instead of one exact number per slot, it writes a best guess plus how unsure it is (a mean and a variance).
- The **decoder** never sees the original drawing. It only reads the sticky note and tries to redraw the cat.
- A penalty (the **KL term**) keeps all the sticky notes written in a similar "language" (close to a standard normal distribution). Because of that shared language, you can later hand the decoder a *random* note and it will still draw something that looks like a cat.

That is the whole trick: reconstruction teaches the decoder to draw, and the KL penalty keeps the latent space tidy enough to sample from.

---

## Results

- **Dataset:** 123,202 cat drawings available; 60,000 used (55,000 train / 5,000 validation)
- **Training:** 20 epochs, NAdam optimizer, learning rate `1e-3`
- **Validation RMSE:** improved from **0.287** (epoch 1) to **0.255** (epoch 20)

| Epoch | Train loss | Train RMSE | Valid RMSE |
|:-----:|:----------:|:----------:|:----------:|
| 1     | 0.0984     | 0.306      | 0.287      |
| 5     | 0.0803     | 0.263      | 0.261      |
| 10    | 0.0787     | 0.257      | 0.257      |
| 20    | 0.0782     | 0.256      | 0.255      |

The notebook produces two visualizations:

1. **Reconstructions:** original validation drawings (top row) vs. the VAE's reconstructions (bottom row).
2. **Generated cats:** a 3×7 grid of new drawings decoded from random latent vectors sampled from `N(0, I)`.

<!-- Save the notebook's output images into an `assets/` folder and uncomment these:
![Reconstructions](assets/reconstructions.png)
![Generated cats](assets/generated_cats.png)
-->

---

## Model Architecture

A small fully connected VAE with a 32-dimensional latent space.

```
Encoder:  Flatten → Linear(784, 128) → ReLU → Linear(128, 64)   # 64 = mean (32) + logvar (32)
Sampler:  z = mean + exp(0.5 · logvar) · ε,   ε ~ N(0, I)       # reparameterization trick
Decoder:  Linear(32, 128) → ReLU → Linear(128, 784) → Sigmoid → Unflatten(1, 28, 28)
```

### Loss function

```
loss = MSE(reconstruction, input) + kl_weight · mean(KL divergence) / 784
```

- **Reconstruction term:** mean squared error between the output and the input image.
- **KL term:** `-0.5 · Σ(1 + logvar − exp(logvar) − mean²)`, averaged over the batch and divided by 784 so it sits on a comparable scale to the per-pixel MSE.
- `kl_weight` defaults to `1.0`.

---

## Training Setup

| Setting | Value |
|---|---|
| Latent dimension | 32 |
| Batch size | 128 |
| Optimizer | NAdam (`lr=1e-3`) |
| LR scheduler | `ReduceLROnPlateau` (mode `max`, patience 2, factor 0.5) |
| Gradient clipping | max norm 1.0 |
| Epochs | 20 |
| Metric | RMSE (`torchmetrics.MeanSquaredError(squared=False)`) |
| Seed | 42 |

---

## Getting Started

### Prerequisites

- Python 3.12 (the notebook was run on 3.12.0)
- A CUDA-capable GPU (the notebook moves the model and data to `"cuda"`)
- Internet access on first run, to download the dataset (~100 MB)

### Installation

```bash
pip install torch torchmetrics numpy matplotlib jupyter
```

### Run

```bash
jupyter notebook vae.ipynb
```

Then run all cells in order. The notebook will:

1. Define the `VAE` class and the loss function
2. Download `cat.npy` from the Quick, Draw! public bucket and build the data loaders
3. Train the model for 20 epochs
4. Plot reconstructions and generate new cat drawings

> **No GPU?** Replace every `"cuda"` in the notebook with `"cpu"` (or `torch.device("cuda" if torch.cuda.is_available() else "cpu")`). Training will be slower but works.

---

## Project Structure

```
.
├── vae.ipynb     # Model, loss, data loading, training loop, and visualizations
└── README.md
```

### Notebook outline

| Section | What it does |
|---|---|
| Create the VAE class | Encoder, reparameterized sampling, decoder, `VAEOutput` namedtuple |
| Define the VAE loss function | MSE reconstruction + weighted KL divergence |
| Load the data | Downloads Quick, Draw! cats, scales to [0, 1], splits train/validation |
| Define the training function | Generic `train` / `evaluate_tm` loop with LR scheduling and gradient clipping |
| Train the VAE | 20 epochs with RMSE tracking |
| Visualize the reconstructions | Side-by-side originals vs. reconstructions |
| Generate cat images | Decodes random latent vectors into new drawings |

---

## Ideas for Extension

- Swap the MLP for a **convolutional** encoder/decoder for sharper drawings
- Sweep `kl_weight` (a **β-VAE**) to trade off reconstruction quality against latent-space smoothness
- Try other Quick, Draw! categories, or train on several at once
- Interpolate between two latent codes to morph one cat into another
- Experiment with different latent sizes (`codings_dim`)

---
