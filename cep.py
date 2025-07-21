import os
import json
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tqdm import tqdm
import scipy.signal as signal
import logging
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("eeg_processing.log"), logging.StreamHandler()]
)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Uncomment for CPU

np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = r"J:\EEG\DBAS22_DataOnline"
OUTPUT_DIR = "testfiles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    "eeg_data_keys": ["signal", "timeseries", "data", "eeg"],
    "label_keys": ["labels", "movement", "target", "class"],
    "target_key": "targets",
    "rom_columns": ["angle", "value", "min", "max"]
}


def inspect_json(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logging.info(f"File: {file_path}, Size: {os.path.getsize(file_path)} bytes")
        logging.info(f"Keys: {list(data.keys())}")
        for key, value in data.items():
            logging.info(f"Key: {key}, Type: {type(value)}, Sample: {str(value)[:100]}...")
        return data
    except Exception as e:
        logging.error(f"Error inspecting {file_path}: {e}")
        return None


def inspect_csv(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        logging.info(f"CSV File: {file_path}, Columns: {list(df.columns)}")
        logging.info(f"Sample Data:\n{df.head().to_string()}")
        return df
    except Exception as e:
        logging.error(f"Error inspecting {file_path}: {e}")
        return None


def wavelet_denoising(data, wavelet="db4", level=3):
    try:
        coeffs = pywt.wavedec(data, wavelet, level=level)
        threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(data)))
        coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
        denoised_data = pywt.waverec(coeffs, wavelet)
        return denoised_data[:len(data)]
    except Exception as e:
        logging.error(f"Error in wavelet denoising: {e}")
        return data


def plot_denoising(raw_data, denoised_data, title="Raw vs Denoised Signal", participant="s1"):
    plt.figure(figsize=(10, 5))
    plt.plot(raw_data[:500], label="Raw Signal", alpha=0.5)
    plt.plot(denoised_data[:500], label="Denoised Signal", linewidth=2)
    plt.title(f"{title} (Participant {participant})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(f"denoising_plot_{participant}.png")
    plt.close()


def extract_features(data, fs=100):
    try:
        mean = np.mean(data)
        variance = np.var(data)
        freqs, psd = signal.welch(data, fs=fs, nperseg=min(256, len(data)))
        dominant_freq = freqs[np.argmax(psd)]
        psd_power = np.sum(psd)
        plt.figure(figsize=(10, 5))
        plt.semilogy(freqs, psd)
        plt.title("Power Spectral Density")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.savefig("psd_plot.png")
        plt.close()
        return np.array([mean, variance, dominant_freq, psd_power])
    except Exception as e:
        logging.error(f"Error in feature extraction: {e}")
        return np.zeros(4)


def augment_data(data, rom_value, num_samples=100):
    base_samples = 100
    extra_samples = 100 if rom_value >= 0.7 else 0
    total_samples = base_samples + extra_samples
    augmented = []
    for _ in range(total_samples):
        noise = np.random.normal(0, 0.03 * np.std(data), len(data))  # Reduced noise
        scaling_factor = rom_value + np.random.uniform(-0.05, 0.05)  # Reduced variability
        scaled_data = data * scaling_factor
        shift = np.random.randint(-10, 10)  # Reduced shift range
        shifted_data = np.roll(scaled_data + noise, shift)
        augmented.append(shifted_data)
    return np.array(augmented, dtype=np.float32)


def build_cnn_transformer(input_shape, num_classes=11):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(128, 3, activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Reduced dropout
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    attention = layers.MultiHeadAttention(8, 256)(x, x)  # Increased heads
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def movement_score(data, rom_min, rom_max):
    try:
        normalized_data = (data - rom_min) / (rom_max - rom_min + 1e-6)
        smoothness = np.std(np.diff(normalized_data))
        amplitude = np.mean(np.abs(normalized_data))
        score = 0.7 * amplitude + 0.3 * (1 - smoothness)
        return np.clip(np.round(score, 1), 0, 1.0)
    except Exception as e:
        logging.error(f"Error in movement scoring: {e}")
        return 0.0


def load_data(participant, phase="initial_acquisition"):
    phase_path = os.path.join(DATA_PATH, f"s{participant}", f"{phase}.json")
    targets_path = os.path.join(DATA_PATH, f"s{participant}", f"targets_{phase}.json")
    rom_path = os.path.join(DATA_PATH, f"s{participant}", "range_of_motion.csv")

    if not os.path.exists(phase_path):
        logging.warning(f"Phase file missing: {phase_path}. Using synthetic data.")
        signal_data = np.random.randn(500).astype(np.float32)
        rom_score = np.random.uniform(0, 1.0)
        rom_score = np.round(rom_score, 1)
        labels = np.full(500, int(rom_score * 10), dtype=np.int32)
    else:
        data = inspect_json(phase_path)
        if data is None:
            signal_data = np.random.randn(500).astype(np.float32)
            rom_score = np.random.uniform(0, 1.0)
            rom_score = np.round(rom_score, 1)
            labels = np.full(500, int(rom_score * 10), dtype=np.int32)
        else:
            for key in CONFIG["eeg_data_keys"]:
                if key in data:
                    signal_data = np.array(data[key], dtype=np.float32)
                    logging.info(f"Using EEG data key: {key}")
                    break
            else:
                logging.warning(f"No EEG data key found in {phase_path}. Using synthetic data.")
                signal_data = np.random.randn(500).astype(np.float32)

            for key in CONFIG["label_keys"]:
                if key in data:
                    labels = np.array(data[key], dtype=np.float32)
                    logging.info(f"Using label key: {key}")
                    break
            else:
                targets_data = inspect_json(targets_path)
                if targets_data and CONFIG["target_key"] in targets_data:
                    labels = np.array(targets_data[CONFIG["target_key"]], dtype=np.float32)
                    logging.info(f"Using targets from {targets_path}")
                else:
                    logging.warning(f"No labels found. Using synthetic labels.")
                    rom_score = np.random.uniform(0, 1.0)
                    rom_score = np.round(rom_score, 1)
                    labels = np.full(len(signal_data), rom_score, dtype=np.float32)

    if signal_data.ndim > 1:
        signal_data = signal_data[:, 0]
    labels = np.round(labels, 1)
    labels = np.clip(labels, 0, 1.0) * 10
    labels = labels.astype(np.int32)

    rom_df = inspect_csv(rom_path)
    if rom_df is None:
        logging.warning(f"ROM file missing: {rom_path}. Using defaults.")
        rom_min, rom_max = 0, 1
    else:
        numeric_cols = [col for col in rom_df.columns if any(k in col.lower() for k in CONFIG["rom_columns"])]
        if not numeric_cols:
            logging.warning(f"No numeric columns in {rom_path}. Using defaults.")
            rom_min, rom_max = 0, 1
        else:
            rom_df[numeric_cols] = rom_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            rom_min = rom_df[numeric_cols].min().min()
            rom_max = rom_df[numeric_cols].max().max()
            if np.isnan(rom_min) or np.isnan(rom_max):
                logging.warning(f"NaN values in {rom_path}. Using defaults.")
                rom_min, rom_max = 0, 1

    denoised_data = wavelet_denoising(signal_data)
    if participant == 1:
        plot_denoising(signal_data, denoised_data, participant=f"s{participant}")

    rom_score = movement_score(denoised_data, rom_min, rom_max)
    labels = np.full(len(denoised_data), int(rom_score * 10), dtype=np.int32)

    return denoised_data[:500], labels[:500], rom_min, rom_max


class tqdm_callback(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_bar = None

    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=self.params['epochs'], desc="Training Epochs", colour='green')

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.update(1)
        self.epoch_bar.set_postfix(loss=logs.get('loss', 0), accuracy=logs.get('accuracy', 0))

    def on_train_end(self, logs=None):
        self.epoch_bar.close()


early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,  # Increased patience
    restore_best_weights=True
)


def main():
    all_data = []
    all_labels = []
    all_features = []
    all_rom_scores = []

    for pid in tqdm(range(1, 21), desc="Processing participants"):
        try:
            data, labels, rom_min, rom_max = load_data(pid)
            features = extract_features(data)
            score = movement_score(data, rom_min, rom_max)

            all_data.append(data)
            all_labels.append(labels)
            all_features.append(features)
            all_rom_scores.append(score)
        except Exception as e:
            logging.warning(f"Skipping participant s{pid} due to error: {e}")
            continue
        finally:
            gc.collect()

    if not all_data:
        raise ValueError("No data processed successfully. Check eeg_processing.log.")

    all_data = np.concatenate([d[:500] for d in all_data], axis=0)
    all_labels = np.concatenate([l[:500] for l in all_labels], axis=0)
    all_features = np.array(all_features)
    all_rom_scores = np.array(all_rom_scores)

    if np.any(np.isnan(all_data)) or np.any(np.isnan(all_labels)):
        logging.error("NaN values detected in data or labels")
        raise ValueError("Invalid data detected. Check eeg_processing.log.")

    logging.info(f"Combined data shape: {all_data.shape}, Labels shape: {all_labels.shape}")

    df = pd.DataFrame(all_features, columns=["mean", "variance", "dominant_freq", "psd_power"])
    df["label"] = all_labels[:len(all_features)]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("feature_correlation.png")
    plt.close()

    rom_values = np.linspace(0.0, 1.0, 11)
    augmented_data = []
    augmented_labels = []
    rom_scores = []

    for i, rom in enumerate(tqdm(rom_values, desc="Augmenting ROM values")):
        aug_data = augment_data(all_data, rom)
        aug_labels = np.full(len(aug_data), i, dtype=np.int32)
        score = movement_score(aug_data.mean(axis=0), np.min(all_data), np.max(all_data))

        filename = os.path.join(OUTPUT_DIR, f"testfile_{i}.npy")
        np.save(filename, aug_data)
        logging.info(f"Saved augmented file: {filename}")

        augmented_data.append(aug_data)
        augmented_labels.extend(aug_labels)
        rom_scores.extend([score] * len(aug_data))

    augmented_data = np.concatenate(augmented_data, axis=0)
    augmented_labels = np.array(augmented_labels)
    rom_scores = np.array(rom_scores)

    logging.info(f"Augmented data shape: {augmented_data.shape}, Labels shape: {augmented_labels.shape}")

    X = augmented_data[:, :, np.newaxis]
    y = augmented_labels

    X_train_val, X_test, y_train_val, y_test, rom_train_val, rom_test = train_test_split(
        X, y, rom_scores, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val, rom_train, rom_val = train_test_split(
        X_train_val, y_train_val, rom_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    logging.info(f"Training data shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    gc.collect()

    logging.info("Building and training CNN-Transformer model...")
    class_weights = {i: 1.0 + (1.0 - i / 10) * 1.0 if i in [1, 4, 6, 9] else 1.0 for i in
                     range(11)}  # Higher weight for 0.1, 0.4, 0.6, 0.9
    model = build_cnn_transformer(input_shape=(X.shape[1], 1))
    history = model.fit(
        X_train, y_train, epochs=75, batch_size=16, validation_data=(X_val, y_val),
        verbose=0, callbacks=[tqdm_callback(), early_stopping], class_weight=class_weights
    )

    model.save("motion_detection_model.keras")
    logging.info("Model saved as motion_detection_model.keras")

    logging.info("Evaluating model...")
    y_val_pred_probs = model.predict(X_val, batch_size=16, verbose=0)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    logging.info(f"Validation predictions shape: {y_val_pred.shape}, True labels shape: {y_val.shape}")

    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

    logging.info(f"Validation Accuracy: {accuracy:.4f}")
    logging.info(f"Validation Precision: {precision:.4f}")
    logging.info(f"Validation Recall: {recall:.4f}")
    logging.info(f"Validation F1-Score: {f1:.4f}")

    class_accuracies = []
    for i in range(11):
        mask = y_val == i
        if np.any(mask):
            class_acc = np.mean(y_val_pred[mask] == y_val[mask])
            class_accuracies.append(class_acc)
            logging.info(f"Class {i / 10:.1f} Accuracy: {class_acc:.4f}")

    y_test_pred_probs = model.predict(X_test, batch_size=16, verbose=0)
    y_test_pred = np.argmax(y_test_pred_probs, axis=1)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Test Precision: {test_precision:.4f}")
    logging.info(f"Test Recall: {test_recall:.4f}")
    logging.info(f"Test F1-Score: {test_f1:.4f}")

    cm = confusion_matrix(y_test, y_test_pred, labels=range(11))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[f"{x / 10:.1f}" for x in range(11)],
                yticklabels=[f"{x / 10:.1f}" for x in range(11)])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted ROM")
    plt.ylabel("True ROM")
    plt.savefig("confusion_matrix.png")
    plt.close()
    logging.info("Saved confusion_matrix.png")

    plt.figure(figsize=(10, 8))
    for i in range(11):
        fpr, tpr, _ = roc_curve(y_test == i, y_test_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROM {i / 10:.1f} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Multi-Class ROC Curve (One-vs-Rest)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()
    logging.info("Saved roc_curve.png")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("training_history.png")
    plt.close()
    logging.info("Saved training_history.png")


if __name__ == "__main__":
    main()