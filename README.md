# Leiden 3D UMAP Visualizer

**Leiden clustering** and **2D/3D UMAP visualization** of single‑cell (or any high‑dimensional) data.

---

## Features
- End‑to‑end workflow in **`Leiden.ipynb`**: load data → preprocess → PCA → kNN graph → **Leiden** clusters → **UMAP (2D & 3D)**.
- Rotating **3D UMAP GIF**.
  
---

## Quick Start

### Create environment & install dependencies
```bash
# Python 3.10+ is recommended
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows note (igraph/leidenalg):** If installation fails on Windows, use Conda:
> ```bash
> conda create -n scenv python=3.10 -y
> conda activate scenv
> conda install -c conda-forge scanpy python-igraph leidenalg umap-learn -y
> pip install imageio Pillow plotly
> ```

### Result

![3D UMAP rotation](assets/umap3d.gif)
![3D UMAP rotation](assets/umap3d.gif)
