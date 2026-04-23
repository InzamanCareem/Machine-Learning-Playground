# 🧪 Machine Learning Playground

An interactive playground for experimenting with machine learning models and **visualizing how hyperparameters affect training behavior**.

This project is designed for exploration, intuition-building, and rapid experimentation — not production use.

---

## ✨ Features

- 🎛️ Interactive hyperparameter controls  
- 📉 Real-time training & validation loss visualization  
- 🔁 Quick parameter sweeps (manual or automated)  
- 📊 Run comparison and history tracking  
- ⚡ Responsive UI with background training  
- 🧩 Modular model + training pipeline  

---

## 🧠 Purpose

This playground helps you build intuition for:

- Learning rate effects on convergence
- Optimizer behavior differences
- Loss function characteristics
- Overfitting vs underfitting
- Training stability and divergence

Instead of relying purely on theory, you can **observe behavior directly**.

---

## 🏗️ Project Structure

```text
.
├── app/                # UI / visualization layer
├── models/             # Model definitions
├── training/           # Training loop + utilities
├── data/               # Dataset loading + preprocessing
├── experiments/        # Run tracking (optional)
└── README.md
```

---

## ⚙️ How It Works

### 1. Data Pipeline
- Load dataset
- Train/test split
- Preprocessing (e.g., normalization, scaling)
- Conversion to model-compatible format

### 2. Model
- Pluggable architecture (e.g., linear model, MLP, etc.)
- Easy to extend or swap

### 3. Training Loop
- Iterative training over epochs
- Tracks metrics such as:
  - Training loss
  - Validation/test loss
- Returns full history for visualization

### 4. Experiment Flow
1. Select hyperparameters  
2. Train model  
3. Visualize results  
4. Compare with previous runs  

---

## 🎛️ Tunable Parameters

Typical parameters you can experiment with:

| Category        | Examples |
|----------------|---------|
| Optimization   | Learning rate, optimizer type |
| Loss Function  | MSE, MAE, Cross-Entropy, etc. |
| Model          | Depth, width, activation |
| Training       | Epochs, batch size |
| Regularization | Dropout, weight decay |

---

## 📊 Visualization

### Training Curves
- Loss vs epochs
- Train vs validation comparison

### Run Comparison
- Overlay multiple runs
- Identify patterns and differences

### Parameter Impact
- Observe how changes affect:
  - Convergence speed
  - Stability
  - Generalization

---

## 🚀 Getting Started

### 1. Install uv (if you don’t have it)

```bash
pip install uv
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Run the application

```bash
uv run python app.py
```

---

## 🧪 Suggested Experiments

- Increase learning rate → observe instability  
- Decrease learning rate → observe slow convergence  
- Compare optimizers (e.g., SGD vs Adam)  
- Switch loss functions → compare sensitivity  
- Modify model size → observe overfitting  

---

## 🧩 Extending the Playground

You can expand this project with:

- Additional models (CNNs, RNNs, Transformers)
- Automated hyperparameter sweeps
- Experiment logging & persistence
- Dataset selection
- Advanced metrics (accuracy, F1, etc.)
- Integration with tools like TensorBoard

---

## ⚠️ Limitations

- Not optimized for large-scale training  
- May lack experiment persistence  
- Simplified training setup  
- Focused on learning and visualization  

---

## 💡 Philosophy

This project prioritizes:

- Simplicity over abstraction  
- Visualization over metrics overload  
- Exploration over automation  

The goal is to make machine learning behavior **intuitive and observable**.

---

## 📜 License

MIT