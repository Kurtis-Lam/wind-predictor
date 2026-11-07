import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import os
import re
from zoneinfo import ZoneInfo
import warnings

# --- Fix for Numpy Deprecation Warning ---
try:
    from numpy._core import multiarray
except ImportError:
    from numpy.core import multiarray

torch.serialization.add_safe_globals([
    multiarray._reconstruct,
    np.ndarray
])
# -----------------------------------------

# --- UPDATED MODEL ARCHITECTURE TO MATCH TRAINING CODE ---
class WindLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        
        # Match Training: bidirectional=True
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True  # <--- CRITICAL FIX: Must match training
        )
        
        # Match Training: Input is hidden*2, and includes BatchNorm
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, 128), # *2 because of bidirectional
            torch.nn.BatchNorm1d(128),             # <--- CRITICAL FIX: Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),                # 0.3 / 2 (from training code)
            torch.nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        
        # Take the output of the last time step
        out = out[:, -1, :] 
        
        return self.fc(out)
# ---------------------------------------------------------

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
        print("❌ No data files found in ./data. Run fetch_data.py first.")
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

def find_model_directories():
    model_dirs = [d for d in os.listdir() if d.startswith("model_") and os.path.isdir(d)]
    if not model_dirs:
        return None
    model_info = []
    for dir_name in model_dirs:
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', dir_name)
        if match:
            timestamp = match.group(1)
            model_info.append((dir_name, timestamp))
        else:
            model_info.append((dir_name, "Unknown timestamp"))
    return model_info

def get_wind_direction_info(u_wind, v_wind):
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    wind_dir_rad = np.arctan2(-u_wind, -v_wind)
    wind_dir_deg = np.degrees(wind_dir_rad) % 360
    
    compass_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                          'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    direction_index = int((wind_dir_deg + 11.25) % 360 / 22.5)
    compass_dir = compass_directions[direction_index]
    
    return wind_speed, wind_dir_deg, compass_dir

def choose_model_directory():
    model_info = find_model_directories()
    if not model_info:
        print("❌ No model directories found. Train a model first.")
        return None
    
    print("\nAvailable models:")
    for i, (dir_name, timestamp) in enumerate(model_info, 1):
        print(f"{i}. {dir_name} ({timestamp})")
    
    while True:
        try:
            selection = input("\nEnter the number of the model to use: ").strip()
            if selection.isdigit():
                index = int(selection) - 1
                if 0 <= index < len(model_info):
                    checkpoint_dir, timestamp = model_info[index]
                    return checkpoint_dir
                else:
                    print(f"❌ Please enter a number between 1 and {len(model_info)}")
            else:
                 print("❌ Please enter a valid number.")
        except ValueError:
            print("❌ Invalid input. Please try again.")

def add_time_features(df):
    """Recreate time features used in training"""
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    return df

def load_model_and_data(checkpoint_dir):
    print(f"⚠️  Using model directory: {checkpoint_dir}")
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if "best" in f.lower() and f.endswith(".pth")]
    if not checkpoint_files:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        
    if not checkpoint_files:
        print("❌ No checkpoint found in directory.")
        return None
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
    print(f"⚠️  Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=torch.device('cpu'),
            weights_only=False
        )
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None
    
    feature_cols = checkpoint.get('feature_cols')
    target_cols = checkpoint.get('target_cols')

    if not feature_cols or not target_cols:
        print("❌ Checkpoint corrupted: Missing feature/target column names.")
        return None

    # --- RECOVERY LOGIC START ---
    if 'target_means' in checkpoint:
        feature_means = checkpoint['feature_means']
        feature_stds = checkpoint['feature_stds']
        target_means = checkpoint['target_means']
        target_stds = checkpoint['target_stds']
    elif 'feature_means' in checkpoint:
        print("⚠️  'target_means' missing. Attempting smart recovery...")
        
        full_means = np.array(checkpoint['feature_means'])
        full_stds = np.array(checkpoint['feature_stds'])
        
        try:
            # Find where targets (u_wind, v_wind) sit in the feature list
            target_indices = [feature_cols.index(col) for col in target_cols]
            print(f"✅ Found targets {target_cols} inside feature list at indices {target_indices}")
            
            feature_means = full_means
            feature_stds = full_stds
            target_means = full_means[target_indices]
            target_stds = full_stds[target_indices]
            
        except ValueError:
            expected_full_len = len(feature_cols) + len(target_cols)
            if len(full_means) == expected_full_len:
                print("✅ Recovered via appending logic.")
                feature_means = full_means[:len(feature_cols)]
                feature_stds = full_stds[:len(feature_cols)]
                target_means = full_means[len(feature_cols):]
                target_stds = full_stds[len(feature_cols):]
            else:
                print(f"❌ CRITICAL ERROR: Could not map targets to saved statistics.")
                return None
    else:
        print("❌ Checkpoint missing all scaling statistics.")
        return None
    # --- RECOVERY LOGIC END ---
    
    hidden_size = checkpoint.get('hidden_size', 128)
    num_layers = checkpoint.get('num_layers', 3)
    
    # Instantiate Model
    model = WindLSTM(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=len(target_cols)
    )
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"\n❌ FATAL ARCHITECTURE MISMATCH: {e}")
        print("   The model defined in this script does not match the model in the checkpoint.")
        return None

    model.eval()
    print(f"✅ Model loaded successfully: input={len(feature_cols)}, output={len(target_cols)}")

    DATA_FILE = choose_data_file()
    if DATA_FILE is None:
        return None

    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded wind data from {DATA_FILE}")
    
    # Preprocessing
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['time'] = df['time'].dt.tz_convert('Asia/Hong_Kong')
    df.set_index('time', inplace=True)

    # ADD THE CYCLICAL FEATURES (Missing in previous script but needed for model)
    df = add_time_features(df)

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        print(f"❌ Missing features in data: {missing_features}")
        return None

    if 'latitude' in df.columns and 'longitude' in df.columns:
        pred_lat = float(df['latitude'].iloc[0])
        pred_lon = float(df['longitude'].iloc[0])
    else:
        pred_lat = 22.3193
        pred_lon = 114.1694

    return {
        'model': model,
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'target_means': target_means,
        'target_stds': target_stds,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'df': df,
        'pred_lat': pred_lat,
        'pred_lon': pred_lon
    }

def predict_one_hour(resources):
    model = resources['model']
    df = resources['df']
    feature_means = resources['feature_means']
    feature_stds = resources['feature_stds']
    target_means = resources['target_means']
    target_stds = resources['target_stds']
    feature_cols = resources['feature_cols']
    pred_lat = resources['pred_lat']
    pred_lon = resources['pred_lon']

    while True:
        pred_time_str = input("\nEnter prediction time (YYYY-MM-DD HH:MM): ")
        try:
            naive_time = datetime.strptime(pred_time_str, "%Y-%m-%d %H:%M")
            pred_time = naive_time.replace(tzinfo=ZoneInfo('Asia/Hong_Kong'))
            # Use 96 hours (sequence length from training)
            seq_len = 96
            start_time = pred_time - timedelta(hours=seq_len)
            end_time = pred_time - timedelta(hours=1)
            
            sequence = df.loc[start_time:end_time]
            
            if len(sequence) != seq_len:
                print(f"⚠️  Found {len(sequence)} hours (need {seq_len}).") 
                print(f"    Available data: {df.index.min()} to {df.index.max()}")
                continue
            break
        except ValueError as ve:
            print(f"❌ Invalid format: {ve}. Use YYYY-MM-DD HH:MM")

    X = sequence[feature_cols].values
    X_scaled = (X - feature_means) / feature_stds
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy().flatten()
    
    pred_unscaled = pred_scaled * target_stds + target_means
    u_pred, v_pred = pred_unscaled
    
    pred_wind_speed, pred_wind_dir_deg, pred_compass_dir = get_wind_direction_info(u_pred, v_pred)
    
    # Retrieve Actual Data if available
    try:
        if pred_time in df.index:
            actual_data = df.loc[pred_time]
            u_actual = actual_data['u_wind_100m']
            v_actual = actual_data['v_wind_100m']
            actual_wind_speed, actual_wind_dir_deg, actual_compass_dir = get_wind_direction_info(u_actual, v_actual)
            has_actual_data = True
        else:
            raise KeyError
    except KeyError:
        print(f"⚠️  No actual data available for {pred_time}")
        has_actual_data = False
    
    if has_actual_data:
        speed_accuracy = 1 - abs(pred_wind_speed - actual_wind_speed) / actual_wind_speed if actual_wind_speed != 0 else 0.0
        direction_diff = abs(pred_wind_dir_deg - actual_wind_dir_deg)
        direction_diff = min(direction_diff, 360 - direction_diff)
        direction_accuracy = 1 - (direction_diff / 180.0)
        total_accuracy = (speed_accuracy + direction_accuracy) / 2
        
        accuracy_text = (f"ACCURACY\n"
                        f"Speed: {speed_accuracy:.2f}\n"
                        f"Direction: {direction_accuracy:.2f}\n"
                        f"Total: {total_accuracy:.2f}")
    else:
        accuracy_text = "Accuracy: N/A"

    print("\n✅ Prediction successful! Generating map...")
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    pad = 0.4
    ax.set_extent([pred_lon - pad, pred_lon + pad, pred_lat - pad, pred_lat + pad], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)
    
    # Plot Location
    ax.plot(pred_lon, pred_lat, 'ro', markersize=10, transform=ccrs.Geodetic())
    
    # Draw Arrow
    scale_factor = 0.1
    arrow_start_lon = pred_lon + np.sin(np.radians(pred_wind_dir_deg)) * scale_factor
    arrow_start_lat = pred_lat + np.cos(np.radians(pred_wind_dir_deg)) * scale_factor
    
    ax.arrow(
        arrow_start_lon, arrow_start_lat,
        pred_lon - arrow_start_lon, pred_lat - arrow_start_lat,
        head_width=0.01, head_length=0.015, fc='red', ec='red',
        length_includes_head=True,
        transform=ccrs.PlateCarree()
    )
    
    prediction_text = (f"PREDICTION for {pred_time_str}\n"
                       f"Wind Speed: {pred_wind_speed:.2f} m/s\n"
                       f"Wind Direction: {pred_compass_dir} ({pred_wind_dir_deg:.1f}°)\n"
                       f"Wind Origin: {get_wind_direction_info(-u_pred, -v_pred)[2]}")
    
    if has_actual_data:
        actual_text = (f"ACTUAL\n"
                       f"Speed: {actual_wind_speed:.2f} m/s\n"
                       f"Direction: {actual_compass_dir} ({actual_wind_dir_deg:.1f}°)")
    else:
        actual_text = "No Actual Data"
    
    plt.figtext(0.15, 0.02, prediction_text, ha="center", bbox={"facecolor":"lightgreen", "alpha":0.8, "pad":5})
    plt.figtext(0.5, 0.02, accuracy_text, ha="center", bbox={"facecolor":"yellow", "alpha":0.8, "pad":5})
    plt.figtext(0.85, 0.02, actual_text, ha="center", bbox={"facecolor":"lightblue", "alpha":0.8, "pad":5})
    
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"prediction_{pred_time.strftime('%Y%m%d_%H%M')}.png"), bbox_inches='tight')
    plt.show()

def predict_period(resources):
    model = resources['model']
    df = resources['df']
    feature_means = resources['feature_means']
    feature_stds = resources['feature_stds']
    target_means = resources['target_means']
    target_stds = resources['target_stds']
    feature_cols = resources['feature_cols']

    while True:
        start_str = input("\nEnter start time (YYYY-MM-DD HH:MM): ")
        end_str = input("Enter end time (YYYY-MM-DD HH:MM): ")
        try:
            start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo('Asia/Hong_Kong'))
            end_time = datetime.strptime(end_str, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo('Asia/Hong_Kong'))
            if end_time <= start_time:
                print("❌ End time must be after start time.")
                continue
            break
        except ValueError:
            print(f"❌ Invalid format.")

    times = pd.date_range(start=start_time, end=end_time, freq="h")
    records = []
    seq_len = 96  # Using 96 from training config

    print("\n⏳ Generating predictions...")
    
    for t in times:
        seq_start = t - timedelta(hours=seq_len)
        seq_end = t - timedelta(hours=1)
        sequence = df.loc[seq_start:seq_end]
        
        if len(sequence) != seq_len:
            continue

        X = sequence[feature_cols].values
        X_scaled = (X - feature_means) / feature_stds
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = model(X_tensor).numpy().flatten()

        pred_unscaled = pred_scaled * target_stds + target_means
        u_pred, v_pred = pred_unscaled
        pred_speed, pred_dir_deg, _ = get_wind_direction_info(u_pred, v_pred)

        # Accuracy Calculation Logic
        if t in df.index:
            u_act = df.loc[t, 'u_wind_100m']
            v_act = df.loc[t, 'v_wind_100m']
            act_speed, act_dir_deg, _ = get_wind_direction_info(u_act, v_act)
            
            # 1. Speed Accuracy
            if act_speed != 0:
                speed_acc = 1 - abs(pred_speed - act_speed) / act_speed
            else:
                speed_acc = 0.0
            
            # 2. Direction Accuracy
            diff = abs(pred_dir_deg - act_dir_deg)
            diff = min(diff, 360 - diff)
            dir_acc = 1 - (diff / 180.0)
            
            # 3. Total Accuracy
            total_acc = (speed_acc + dir_acc) / 2
        else:
            act_speed = act_dir_deg = np.nan
            speed_acc = dir_acc = total_acc = np.nan

        records.append({
            'time': t,
            'speed_pred': pred_speed,
            'dir_deg_pred': pred_dir_deg,
            'speed_act': act_speed,
            'dir_deg_act': act_dir_deg,
            'speed_acc': speed_acc,
            'dir_acc': dir_acc,
            'total_acc': total_acc
        })

    if not records:
        print("❌ No predictions possible (insufficient data history).")
        return

    result_df = pd.DataFrame(records).set_index('time')

    # --- PRINT SUMMARY REPORT ---
    avg_speed_acc = result_df['speed_acc'].mean()
    avg_dir_acc = result_df['dir_acc'].mean()
    avg_total_acc = result_df['total_acc'].mean()

    print("\n" + "="*40)
    print("📊 PERIOD ACCURACY REPORT")
    print("="*40)
    print(f"Time Range: {start_time} to {end_time}")
    print(f"Data Points: {len(result_df)}")
    print("-" * 40)
    print(f"Average Speed Accuracy:     {avg_speed_acc:.2%}")
    print(f"Average Direction Accuracy: {avg_dir_acc:.2%}")
    print(f"Average Total Accuracy:     {avg_total_acc:.2%}")
    print("="*40)
    
    # --- PLOT RESULTS ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Row 1: Speed
    axes[0].plot(result_df.index, result_df['speed_act'], label='Actual', color='blue')
    axes[0].plot(result_df.index, result_df['speed_pred'], label='Predicted', color='orange', linestyle='--')
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].set_title('Wind Speed Comparison')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Row 2: Direction
    axes[1].plot(result_df.index, result_df['dir_deg_act'], label='Actual', color='green')
    axes[1].plot(result_df.index, result_df['dir_deg_pred'], label='Predicted', color='red', linestyle='--')
    axes[1].set_ylabel('Direction (°)')
    axes[1].set_title('Wind Direction Comparison')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Row 3: Accuracy
    axes[2].plot(result_df.index, result_df['total_acc'], label='Total Accuracy', color='purple', linewidth=2)
    axes[2].plot(result_df.index, result_df['speed_acc'], label='Speed Acc', color='orange', linestyle=':', alpha=0.7)
    axes[2].plot(result_df.index, result_df['dir_acc'], label='Dir Acc', color='red', linestyle=':', alpha=0.7)
    axes[2].axhline(y=avg_total_acc, color='purple', linestyle='--', alpha=0.5, label=f'Avg Total ({avg_total_acc:.2f})')
    
    axes[2].set_ylabel('Accuracy (0-1)')
    axes[2].set_title('Model Accuracy Over Time')
    axes[2].set_ylim(0, 1.1)
    axes[2].legend(loc='lower right')
    axes[2].grid(True, alpha=0.3)
    
    plt.xlabel('Time')
    plt.tight_layout()
    
    # --- FIX: Ensure directory exists ---
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)  # <--- This line was missing
    
    filename = os.path.join(output_dir, f"period_report_{start_time.strftime('%Y%m%d')}.png")
    plt.savefig(filename)
    print(f"\n✅ Plot saved to {filename}")
    plt.show()

def main():
    print("\n" + "="*50)
    print("🌪️  Wind Prediction System V3")
    print("="*50)

    checkpoint_dir = choose_model_directory()
    if checkpoint_dir is None:
        return

    resources = load_model_and_data(checkpoint_dir)
    if resources is None:
        return

    while True:
        mode = input("\nType 'predict' (single), 'predict_period' (range), or 'quit': ").strip().lower()
        if mode == 'predict':
            predict_one_hour(resources)
        elif mode == 'predict_period':
            predict_period(resources)
        elif mode in ('quit', 'exit', 'q'):
            break
        else:
            print("❌ Invalid command.")

if __name__ == "__main__":
    main()