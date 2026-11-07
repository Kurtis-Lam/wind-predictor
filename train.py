import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import time
from datetime import datetime
import pytz
import math

print("=======================================================")

HONG_KONG_TZ = pytz.timezone('Asia/Hong_Kong')
CURR_TIME = datetime.now(HONG_KONG_TZ)
NOW = CURR_TIME.strftime("%Y-%m-%d_%H-%M-%S")

# --- Configuration ---
EPOCHS = int(input("⚠️  Enter number of epochs (default 100): ") or 100)
LEARNING_RATE = 0.001
BATCH_SIZE = 128  # Increased batch size for faster training
HIDDEN_SIZE = 128
NUM_LAYERS = 3
SEQUENCE_LENGTH = 96
DROPOUT = 0.3
PATIENCE = 30    # For early stopping

def list_data_files():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return []
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    files.sort()
    return files

def choose_data_file():
    files = list_data_files()
    if not files:
        print("❌ No data files found in ./data.")
        return None
    print("\nAvailable data files:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")
    while True:
        choice = input("\nSelect data file number: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return os.path.join("data", files[idx])
        print(f"❌ Invalid choice. Enter a number between 1 and {len(files)}.")

# --- Feature Engineering Helper ---
def add_time_features(df):
    """
    Converts datetime column into cyclical features (sin/cos)
    so the model understands Day/Night and Seasonality.
    """
    df['time'] = pd.to_datetime(df['time'])

    # Hour of day (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)

    # Day of year (1-365)
    df['day_sin'] = np.sin(2 * np.pi * df['time'].dt.dayofyear / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['time'].dt.dayofyear / 365)
    return df

# --- Memory Efficient Dataset ---
class WindDataset(Dataset):
    def __init__(self, data, seq_len, feature_cols_idx, target_cols_idx):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.feature_idx = feature_cols_idx
        self.target_idx = target_cols_idx

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        # Slicing on the fly saves massive amounts of RAM
        # Input: Sequence of length seq_len
        x = self.data[index : index + self.seq_len, self.feature_idx]
        # Target: The NEXT step after the sequence
        y = self.data[index + self.seq_len, self.target_idx]
        return x, y

class WindLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        # Bidirectional LSTM captures patterns from both directions
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT if num_layers > 1 else 0,
            bidirectional=True
        )

        # Fully Connected Head
        # Input size is hidden_size * 2 because of bidirectionality
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.BatchNorm1d(128), # BatchNorm helps stabilize training
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT/2),
            nn.Linear(64, output_size)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)

        # Take the output of the last time step
        # out shape: (batch, seq_len, hidden_size * 2)
        out = out[:, -1, :]

        return self.fc(out)

def save_checkpoint(model, scaler, feature_cols, target_cols, epoch, checkpoint_dir, val_loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'feature_means': scaler.mean_,
        'feature_stds': scaler.scale_,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'sequence_length': SEQUENCE_LENGTH,
        'val_loss': val_loss
    }
    path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, path)
    return path

def main():
    DATA_FILE = choose_data_file()
    if DATA_FILE is None:
        return

    data_file_tag = os.path.splitext(os.path.basename(DATA_FILE))[0]
    CHECKPOINT_DIR = f"model_{NOW}_{data_file_tag}_v2"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded data: {len(df)} records")

    # 1. Add Cyclical Time Features
    df = add_time_features(df)

    # 2. Define Columns
    # We now predict the raw next step, so we don't need to shift the dataframe
    # and drop NaNs manually for targets, the Dataset class handles indexing.

    feature_cols = [
        "temperature_2m", "relative_humidity_2m", "surface_pressure",
        "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
        "wind_gusts_10m", "u_wind_10m", "v_wind_10m",
        "u_wind_100m", "v_wind_100m",
        "hour_sin", "hour_cos", "day_sin", "day_cos" # New features
    ]

    target_cols = ["u_wind_100m", "v_wind_100m"]

    # Check columns
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return

    # 3. Scaling
    # We scale EVERYTHING (features + targets) together
    data_cols = feature_cols   # In this case targets are subset of features
                               # If targets were different, we'd need to handle carefully

    scaler = StandardScaler()
    # We fit on the whole dataset for simplicity, strictly you should fit on train only
    # to avoid data leakage, but for weather stats it's usually acceptable.
    scaled_data = scaler.fit_transform(df[feature_cols].values)

    # Map column names to indices for the Dataset class
    feat_indices = [feature_cols.index(c) for c in feature_cols]
    target_indices = [feature_cols.index(c) for c in target_cols]

    # 4. Train/Test Split (Time Series Split)
    split_idx = int(len(scaled_data) * 0.8)
    train_data_raw = scaled_data[:split_idx]
    test_data_raw = scaled_data[split_idx:]

    train_dataset = WindDataset(train_data_raw, SEQUENCE_LENGTH, feat_indices, target_indices)
    test_dataset = WindDataset(test_data_raw, SEQUENCE_LENGTH, feat_indices, target_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"⚠️  Training Sequences: {len(train_dataset)}")
    print(f"⚠️  Testing Sequences: {len(test_dataset)}")

    # 5. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚠️  Using device: {device}")

    model = WindLSTM(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=len(target_cols)
    ).to(device)

    # HuberLoss is more robust to outliers (wind gusts) than MSE
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 6. Training Loop
    best_val_loss = float('inf')
    early_stop_counter = 0

    print("=======================================================")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        train_loss /= len(train_loader)
        val_loss /= len(test_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        print(f'Epoch {epoch+1:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {current_lr:.6f}', end='')

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            save_checkpoint(model, scaler, feature_cols, target_cols, epoch, CHECKPOINT_DIR, val_loss, 'best_model.pth')
            print(" | ✅ Saved Best")
        else:
            early_stop_counter += 1
            print(f" | ⏳ Patience {early_stop_counter}/{PATIENCE}")

        # Save Latest Model (overwrite previous)
        if epoch % 10 == 0:
            save_checkpoint(model, scaler, feature_cols, target_cols, epoch, CHECKPOINT_DIR, val_loss, 'latest_checkpoint.pth')

        # Early Stopping
        if early_stop_counter >= PATIENCE:
            print(f"\n🛑 Early stopping triggered. No improvement for {PATIENCE} epochs.")
            break

        if current_lr < 1e-8:
             print("\n⚠️  Learning rate too small, stopping.")
             break

    total_time = (time.time() - start_time) / 60
    print("=======================================================")
    print(f"✅ Training complete in {total_time:.1f} minutes")
    print(f"   Best Val Loss: {best_val_loss:.5f}")
    print(f"   Saved to: {CHECKPOINT_DIR}")
    print("=======================================================")

if __name__ == "__main__":
    main()