# DIA-BERT-TimsTOF

A deep learning model for DIA (Data Independent Acquisition) mass spectrometry data analysis using BERT-based architecture for TimsTOF instruments.

## Project Structure

```
DIA-BERT-TimsTOF/
├── calc_fdr.py          # False Discovery Rate calculation
├── dataset.py           # Dataset handling and preprocessing
├── eval.py             # Model evaluation scripts
├── getdata.py          # Data loading and processing utilities
├── model.py            # BERT-based model architecture
├── train.py            # Training pipeline
├── utils.py            # Utility functions
├── mv_pkl.py           # Pickle file management
├── yaml/               # Configuration files
│   └── train_config.yaml
├── checkpoints/        # Model checkpoints (ignored in git)
├── logs/              # Training logs (ignored in git)
├── outputs/           # Model outputs (ignored in git)
├── val_data/          # Validation data
└── fdrs/              # FDR results (ignored in git)
```

## Features

- BERT-based architecture for mass spectrometry data analysis
- Support for TimsTOF DIA data processing
- PyTorch Lightning integration for efficient training
- Comprehensive evaluation metrics and FDR calculation
- Configurable training pipeline with YAML configuration

## Setup

1. Clone the repository:
```bash
git clone https://github.com/AugustSirius/dia-model.git
cd dia-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure training parameters in `yaml/train_config.yaml`

## Usage

### Training
```bash
python train.py
```

### Evaluation
```bash
python eval.py
```

### FDR Calculation
```bash
python calc_fdr.py
```

## Model Architecture

The model uses a BERT-based architecture specifically adapted for mass spectrometry data analysis, incorporating:
- Transformer attention mechanisms for peptide sequence analysis
- Specialized embeddings for mass spectrometry features
- Multi-task learning for improved prediction accuracy

## Configuration

Training parameters can be configured in `yaml/train_config.yaml`. Key parameters include:
- Learning rate and optimization settings
- Batch size and data loading parameters
- Model architecture configurations
- Evaluation metrics and logging options

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```
@article{dia-bert-timstof,
  title={DIA-BERT-TimsTOF: BERT-based Model for DIA Mass Spectrometry Data Analysis},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```
