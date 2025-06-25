# Distance-Based Event Classification using 1D ResNet

A deep learning project for binary event classification using time-series distance data. The model achieves **93.1% test accuracy** and **95.9% ROC-AUC** using a custom 1D ResNet architecture.

## 🎯 Results

- **Test Accuracy**: 93.1%
- **ROC-AUC Score**: 95.9%
- **Training Time**: 60 epochs with adaptive learning rate scheduling

## 📊 Dataset

- **Input Data**: Excel file with 170 time-steps × 5 distance features
- **Format**: Each sample represents a time series with 5 distance measurements over 170 time steps
- **Labels**: Binary classification (0/1)
- **Data Split**: 70% training / 15% validation / 15% testing (stratified)

## 🏗️ Model Architecture

### 1D ResNet Features:
- **Input**: (batch_size, 170, 5) → time-series with 5 distance features
- **Stem**: 1D convolution with 7×1 kernel, 64 filters
- **Residual Blocks**: 3 blocks with skip connections
  - Block 1: 64 → 128 channels
  - Block 2: 128 → 256 channels  
  - Block 3: 256 → 256 channels
- **Pooling**: Adaptive average pooling
- **Output**: Single logit for binary classification

### Key Components:
- Batch normalization for stable training
- ReLU activations
- Skip connections for gradient flow
- Global average pooling to prevent overfitting

## 🚀 Training Configuration

```python
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
OPTIMIZER = Adam (weight_decay=1e-4)
LOSS_FUNCTION = BCEWithLogitsLoss
SCHEDULER = ReduceLROnPlateau (factor=0.5, patience=5)
```

## 📈 Training Progress

The model shows excellent convergence with adaptive learning rate scheduling:

- **Initial LR**: 1e-3 (epochs 1-11)
- **Reduced to**: 5e-4 (epochs 12-42) 
- **Further reduced to**: 2.5e-4 (epochs 43-48)
- **Final reductions**: 1.25e-4 → 6.25e-5

**Training Progression:**
- Epoch 1: 70.7% train acc, 61.4% val acc
- Epoch 30: 95.5% train acc, 92.1% val acc  
- Epoch 60: 97.0% train acc, 95.0% val acc

## 📁 Project Structure

```
Sleep_Apnea/
├── main.py                           # Main training script
├── main.ipynb                        # Jupyter notebook version
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── Data_Prep/
    ├── CNN_Input_Labeled_0_1_only.xlsx    # Main dataset
    ├── Labels Three Events.csv             # Label data
    ├── Multiple Distances Three Events.csv # Distance measurements
    └── merging_excels.py                   # Data preprocessing script
```

## 🛠️ Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy torch scikit-learn matplotlib openpyxl
```

### Dependencies:
- `pandas>=1.5.0` - Data manipulation and Excel file reading
- `numpy>=1.21.0` - Numerical computations
- `torch>=1.12.0` - Deep learning framework
- `scikit-learn>=1.1.0` - Machine learning utilities
- `matplotlib>=3.5.0` - Plotting and visualization
- `openpyxl>=3.0.0` - Excel file support

## 🚀 Usage

### Quick Start:

1. **Clone/Download the project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run training:**
   ```bash
   python main.py
   ```

### Alternative - Jupyter Notebook:
```bash
jupyter notebook main.ipynb
```

### Key Features:
- **Data Normalization**: Global z-score normalization across all features
- **Stratified Splitting**: Maintains class balance across train/val/test sets
- **Early Stopping**: Learning rate reduction on validation loss plateau
- **Confusion Matrix**: Visual evaluation of model performance

## 📊 Model Performance

The model demonstrates excellent performance on the test set:

```
🎯 Test accuracy 0.931 | ROC-AUC 0.959
```

**Confusion Matrix**: Displayed automatically after training completion with color-coded visualization.

## 🔧 Key Implementation Details

### Data Processing:
- Reshapes flat Excel data into (samples, 170, 5) time-series format
- Applies feature-wise z-score normalization
- Uses stratified sampling to maintain class distribution

### Model Training:
- Implements custom residual blocks with 1D convolutions
- Uses batch normalization and skip connections
- Employs adaptive learning rate scheduling
- Monitors validation loss for learning rate reduction

### Evaluation:
- Binary classification with 0.5 threshold
- Reports both accuracy and ROC-AUC metrics
- Generates confusion matrix visualization

## 🎓 Technical Highlights

- **ResNet Architecture**: Adapted for 1D time-series data
- **Skip Connections**: Enable training of deeper networks
- **Adaptive Pooling**: Handles variable sequence lengths
- **BCEWithLogitsLoss**: Numerically stable sigmoid + BCE loss
- **Learning Rate Scheduling**: Automatic reduction on plateau

---

*This project demonstrates effective application of deep learning to time-series classification with excellent results on distance-based event detection.*
