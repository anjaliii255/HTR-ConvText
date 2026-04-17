# HTR-ConvText iMac Runbook
Follow these instructions precisely to set up and train the HTR-ConvText model natively on your Apple Silicon iMac.

## Step 1: Transfer Files
1. Copy the `HTR-ConvText` folder (which we just packaged) to your iMac.
2. Move your downloaded `IAM_lines.zip` and `read2016_lines.zip` files next to the `HTR-ConvText` folder.

## Step 2: Environment Setup
Open the Terminal app on your iMac and navigate to the directory:
```bash
cd path/to/HTR-ConvText
```

Since Conda is not installed, we will use Python's built-in `venv` to create an isolated environment optimized for your Apple Native GPU (MPS/Metal):
```bash
# Create a virtual environment named "htr-env"
python3 -m venv htr-env

# Activate the virtual environment
source htr-env/bin/activate

# Install PyTorch optimized for macOS
pip3 install torch torchvision torchaudio

# Install remaining libraries
pip install -r requirements.txt
```

## Step 3: Dataset Preparation
Run the automated tool I built for you to bypass the bugs in the official repository splits. It will handle unzipping, validating image-text pairs, and 80-10-10 dataset splitting automatically.

```bash
# Point the paths exactly to where your ZIP files are located
python setup_data.py --iam_zip ../IAM_lines.zip --read_zip ../read2016_lines.zip
```

> [!NOTE] 
> This might take a minute depending on the iMac's disk speed. It will automatically populate `data/iam` and `data/read2016` completely flawlessly for training.

## Step 4: Start Training

To train the model on the IAM dataset natively using your iMac's Metal GPU (MPS), use the following command:

```bash
python train.py --dataset iam --tcm-enable --exp-name "imac-training-iam" --img-size 512 64 --train-bs 32 --val-bs 8 --data-path data/iam/lines/ --train-data-list data/iam/train.ln --val-data-list data/iam/val.ln --test-data-list data/iam/test.ln --nb-cls 80
```
*(Notice the batch sizes `train-bs` and `val-bs` have been lowered to 32 and 8 since the Mac Unified Memory might hit bottlenecks compared to a gigantic Nvidia H100 GPU! Feel free to increase them to 64 and 16 if your iMac has 64GB+ of unified memory.)*

To train on READ2016, swap the dataset arguments:
```bash
python train.py --dataset read2016 --tcm-enable --exp-name "imac-training-read2016" --img-size 512 64 --train-bs 32 --val-bs 8 --data-path data/read2016/lines/ --train-data-list data/read2016/train.ln --val-data-list data/read2016/val.ln --test-data-list data/read2016/test.ln --nb-cls 90
```

> [!TIP]
> You will see `Using device: mps` in your terminal logs. This proves the system is officially hardware-accelerated for Apple Silicon!
