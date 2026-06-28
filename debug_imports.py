import sys
print("Python:", sys.version)

print("Step 1: importing os, sys, pathlib...")
import os
from pathlib import Path
print("OK")

print("Step 2: importing numpy, torch...")
import numpy as np
import torch
print("OK")

print("Step 3: importing sklearn...")
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
print("OK")

print("Step 4: importing librosa, psutil...")
import librosa
import psutil
print("OK")

print("Step 5: importing utils...")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), ""))
sys.path.insert(0, ".")
from utils.config   import DATA_ROOT, RESULTS_DIR, MACHINES, SAMPLE_RATE, get_device
from utils.features import wav_paths
from utils.evaluate import evaluate, save_results
print("OK")

print("Step 6: importing panns_inference...")
from panns_inference import AudioTagging
print("OK")

print("Step 7: loading CNN14 model...")
at = AudioTagging(checkpoint_path=None, device="cpu")
print("OK — model loaded")

print("Step 8: listing children...")
for name, _ in at.model.named_children():
    print(f"  {name}")

print("\nAll imports OK — the script should work.")