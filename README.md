# Graph Neural Network for ENZYMES Dataset

## How to Train the Model
1. Run `model.py` to train the model and save checkpoints:
   ```bash
   python model.py
2.	Checkpoints will be saved in the saved_models/ directory.

## How to Test the Model

1.	Open test.py and update the path to the saved model:
    ```bash
    saved_model_path = "gcn_model_142.pth"  # Replace with your saved model (or use this one which is already provided)
2.	Run test.py to evaluate the model:
    ```bash
    python test.py
3.	The script will print Accuracy, F1 Score, and AUC and plot the AUC-ROC curve.

### Dependencies

Install required packages:

    pip install torch torchvision torchaudio torch-geometric scikit-learn matplotlib numpy

### Directory Structure

```bash
.
├── ENZYMES/             # Dataset folder
├── model.py             # Script for training
├── test.py              # Script for testing
├── saved_models/        # Saved model checkpoints

