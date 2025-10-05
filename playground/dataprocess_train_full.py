import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras.layers import LSTM
from tensorflow.keras.layers import Input, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
import datetime
import tensorflow as tf

# --- Data preprocessing ---
fires = pd.read_csv("C:\\Users\\Bence\\Documents\\Github\\NASA_SpaceApps_challenge_2025\\data\\greece_fire_dates.csv")
places = pd.read_csv("C:\\Users\\Bence\\Documents\\Github\\NASA_SpaceApps_challenge_2025\\data\\greece_fire_places.csv")
weather = pd.read_csv("C:\\Users\\Bence\\Documents\\Github\\NASA_SpaceApps_challenge_2025\\data\\greece_fire_weather.csv")

fires = fires.drop(fires[fires["confidence"] < 100].index)
fires["acq_date"] = pd.to_datetime(fires["acq_date"], errors="coerce")
weather["time"] = pd.to_datetime(weather["time"], errors="coerce")

fires_daily = fires.groupby("acq_date").size().reset_index(name="fire_count")
fires_daily["fire_occurred"] = (fires_daily["fire_count"] > 0).astype(int)

places = places.drop(columns=["elevation","utc_offset_seconds","timezone","timezone_abbreviation"], errors="ignore")
weather = weather.merge(places, on="location_id", how="left")

numeric_cols = weather.select_dtypes(include=[np.number]).columns
weather_daily = weather.groupby("time")[numeric_cols].mean().reset_index()

df = weather_daily.merge(fires_daily, left_on="time", right_on="acq_date", how="left")
df["fire_occurred"].fillna(0, inplace=True)
df["fire_count"].fillna(0, inplace=True)
df.drop(columns=["acq_date"], inplace=True, errors="ignore")
df.sort_values("time", inplace=True)
df.reset_index(drop=True, inplace=True)

# --- Windows ---
window_size = 30
forecast_horizon = 1

feature_cols = [
    "temperature_2m_max (°C)",
    "temperature_2m_min (°C)",
    "apparent_temperature_max (°C)",
    "apparent_temperature_min (°C)",
    "daylight_duration (s)",
    "sunshine_duration (s)",
    "precipitation_sum (mm)",
    "rain_sum (mm)",
    "snowfall_sum (cm)",
    "precipitation_hours (h)",
    "wind_speed_10m_max (km/h)",
    "wind_gusts_10m_max (km/h)",
    "wind_direction_10m_dominant (°)",
    "shortwave_radiation_sum (MJ/m²)",
    "et0_fao_evapotranspiration (mm)"
]

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

X, y = [], []
for i in range(len(df_scaled) - window_size - forecast_horizon):
    past_window = df_scaled.iloc[i:i + window_size][feature_cols].values
    target_window = df_scaled.iloc[i + window_size:i + window_size + forecast_horizon]["fire_occurred"].values
    X.append(past_window)
    y.append(target_window)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, forecast_horizon, 1)

print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")

# --- Encoder-decoder LSTM ---
def build_encoder_decoder_lstm(input_shape, forecast_horizon):
    encoder_inputs = Input(shape=input_shape)
    encoder = LSTM(128, activation='tanh', return_sequences=False)(encoder_inputs)
    encoder = Dropout(0.3)(encoder)
    bottleneck = Dense(64, activation='relu')(encoder)
    decoder = RepeatVector(forecast_horizon)(bottleneck)
    decoder = LSTM(128, activation='tanh', return_sequences=True)(decoder)
    decoder = Dropout(0.3)(decoder)
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(decoder)
    model = Model(encoder_inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name="auc")]
    )
    model.summary()
    return model

# --- Downsampling function ---
def downsample(X, y, pos_fraction=0.2, random_state=42):
    np.random.seed(random_state)
    y_any = y.max(axis=1).flatten()  # 1 ha bármelyik nap tűz volt
    pos_idx = np.where(y_any == 1)[0]
    neg_idx = np.where(y_any == 0)[0]

    n_pos = len(pos_idx)
    n_neg = int(n_pos * (1 - pos_fraction) / pos_fraction)

    if n_neg > len(neg_idx):
        n_neg = len(neg_idx)
    sampled_neg_idx = np.random.choice(neg_idx, size=n_neg, replace=False)

    final_idx = np.concatenate([pos_idx, sampled_neg_idx])
    np.random.shuffle(final_idx)

    return X[final_idx], y[final_idx]

# --- Training ---
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

tscv = TimeSeriesSplit(n_splits=5)
y_any = y.max(axis=1).flatten()
unique, counts = np.unique(y_any, return_counts=True)
print("Full dataset distribution:", dict(zip(unique, counts)))

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    X_train_fold, y_train_fold = X[train_idx], y[train_idx]
    X_test_fold, y_test_fold = X[test_idx], y[test_idx]

    # --- Downsampling a train halmaz ---
    X_train_ds, y_train_ds = downsample(X_train_fold, y_train_fold, pos_fraction=0.2)
    print(f"Fold {fold}: train_pos={y_train_ds.max(axis=1).sum()}/{len(y_train_ds)}")

    model = build_encoder_decoder_lstm(
        input_shape=(X_train_ds.shape[1], X_train_ds.shape[2]),
        forecast_horizon=forecast_horizon
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_ds, y_train_ds,
        validation_data=(X_test_fold, y_test_fold),
        epochs=40,
        batch_size=32,
        callbacks=[early_stop, tensorboard_callback],
        verbose=1
    )

    # --- Save model ---
    model.save(f"encoder_decoder_fold{fold}.keras")

    # --- Evaluate ---
    y_pred_seq = model.predict(X_test_fold)
    y_pred_any = y_pred_seq.max(axis=1).flatten()
    y_test_any = y_test_fold.max(axis=1).flatten()

    rocauc = roc_auc_score(y_test_any, y_pred_any)
    prauc = average_precision_score(y_test_any, y_pred_any)
    y_pred_bin = (y_pred_any > 0.5).astype(int)

    print(f"Fold {fold} - ROC-AUC: {rocauc:.4f} | PR-AUC: {prauc:.4f}")
    print(classification_report(y_test_any, y_pred_bin, digits=4))

    # --- Plot ---
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"Fold {fold} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # plt.subplot(1,3,2)
    # plt.plot(history.history["accuracy"], label="Train Acc")
    # plt.plot(history.history["val_accuracy"], label="Val Acc")
    # plt.title(f"Fold {fold} Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.grid(True)

    plt.subplot(1,3,3)
    plt.plot(history.history["auc"], label="Train AUC")
    plt.plot(history.history["val_auc"], label="Val AUC")
    plt.title(f"Fold {fold} AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

print("\nAll folds training complete.")
