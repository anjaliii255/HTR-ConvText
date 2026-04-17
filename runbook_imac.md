# iMac Deployment & Training Runbook

This runbook outlines the exact end-to-end steps required to pull the HTR-ConvText codebase onto your institutional iMac, prepare the datasets locally without syncing gigabytes of data over Git, and effortlessly initialize PyTorch training fully leveraging Apple Silicon (MPS Metal GPU natively). 

---

### Step 1: Clone the Repository & Environment Setup
Log into your iMac terminal and pull the synced codebase. We use Python's built-in `venv` to avoid bloated Conda path issues on campus machines.

```bash
# Clone the synced codebase
git clone https://github.com/anjaliii255/HTR-ConvText.git
cd HTR-ConvText

# Prepare an isolated Python environment
python3 -m venv htr-env
source htr-env/bin/activate

# Install requirements (make sure to use 'pip3' if 'pip' gives errors)
pip install -r requirements.txt
```

### Step 2: Extract & Structure Datasets Automatically
Download `IAM_lines.zip` and `read2016_lines.zip` manually from your Google Drive into the iMac's `~/Downloads` folder. 

Run the automated data processing script. This single script handles the decompression, validation of text pairs, and 80-10-10 file splitting (`train.ln`, `val.ln`, `test.ln`):

```bash
python setup_data.py --iam_zip ~/Downloads/IAM_lines.zip --read_zip ~/Downloads/read2016_lines.zip
```

### Step 3: Train the Models on MPS
PyTorch natively hooks into the M-series GPU for 99% of its deep learning graphs (`Using device: mps`). The `ctc_loss` function is explicitly handled in our `train.py` locally via CPU fallback, so you **do not** need to mess with any environment wrappers.

*(Note: The batch sizes `train-bs` and `val-bs` have been lowered to 32 and 8 since the Mac Unified Memory might hit bottlenecks compared to a server GPU. Feel free to increase them to 64 and 16 if your iMac has 64GB+ of unified memory.)*

**Train on the IAM Dataset:**
```bash
python train.py --dataset iam --tcm-enable --exp-name "imac-training-iam" --img-size 512 64 --train-bs 32 --val-bs 8 --data-path data/iam/lines/ --train-data-list data/iam/train.ln --val-data-list data/iam/val.ln --test-data-list data/iam/test.ln --nb-cls 80
```

**Train on the READ2016 Dataset:**
```bash
python train.py --dataset read2016 --tcm-enable --exp-name "imac-training-read2016" --img-size 512 64 --train-bs 32 --val-bs 8 --data-path data/read2016/lines/ --train-data-list data/read2016/train.ln --val-data-list data/read2016/val.ln --test-data-list data/read2016/test.ln --nb-cls 90
```
