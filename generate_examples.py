import numpy as np
import tensorflow as tf
from tensorflow import keras

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Ground truth function
def f(x):
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1

# Generate dataset
N = 100
xs = np.random.uniform(-2, 2, N)
xs = np.sort(xs)

ys = f(xs)

# Split into train and test
half = N // 2
x_train = xs[:half]
y_train = ys[:half]
x_test = xs[half:half*2]
y_test = ys[half:half*2]

# Add noise
noise_std = np.sqrt(0.05)
y_train_noisy = y_train + np.random.normal(0, noise_std, len(y_train))
y_test_noisy = y_test + np.random.normal(0, noise_std, len(y_test))

# Function to create and train model
def create_and_train(x_train, y_train, x_test, y_test, epochs, name):
    model = keras.Sequential([
        keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), 
                  loss='mse')
    
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, 
                       verbose=0, validation_data=(x_test, y_test))
    
    train_loss = model.evaluate(x_train, y_train, verbose=0)
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    
    return train_loss, test_loss, history

print("=" * 70)
print("FFNN Regression - Overfitting Progression")
print("=" * 70)
print(f"\nDataset: N={N}, Noise Std={noise_std:.4f}")
print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
print("\n" + "=" * 70)

scenarios = [
    ("Early Stop", 10),
    ("Moderate", 30),
    ("Good", 50),
    ("Better", 100),
    ("Slight Overfit", 200),
    ("Clear Overfit", 500),
    ("Strong Overfit", 1000)
]

results = []

for name, epochs in scenarios:
    train_mse, test_mse, _ = create_and_train(x_train, y_train_noisy, 
                                               x_test, y_test_noisy, 
                                               epochs, name)
    gap = train_mse - test_mse  # Negative means test is worse (overfitting)
    ratio = test_mse / train_mse if train_mse > 0 else 0
    
    results.append({
        'name': name,
        'epochs': epochs,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'gap': gap,
        'ratio': ratio
    })
    
    print(f"\n{name:20} | Epochs: {epochs:4d}")
    print(f"  Train MSE: {train_mse:.4e}  |  Test MSE: {test_mse:.4e}")
    print(f"  Gap (Train-Test): {gap:.4e}  |  Test/Train Ratio: {ratio:.4f}")
    
    if gap < -0.001:
        status = "✗ OVERFIT: Test worse than Train"
    elif gap < 0.001:
        status = "≈ BALANCED: Very similar"
    else:
        status = "↑ Normal: Test slightly worse (expected)"
    print(f"  → {status}")

print("\n" + "=" * 70)
print("EMPFOHLENE BEISPIELE FÜR DIE DOKUMENTATION:")
print("=" * 70)

# Select best examples
best_fit = results[2]  # Should be around epoch 50
slight_overfit = results[4]  # Epoch 200
clear_overfit = results[5]  # Epoch 500

print(f"\n### Best-Fit Beispiel:")
print(f"Epochs: {best_fit['epochs']}")
print(f"Train MSE: {best_fit['train_mse']:.3e} | Test MSE: {best_fit['test_mse']:.3e}")
print(f"Gap: {best_fit['gap']:.3e} (minimal)")

print(f"\n### Slight Overfit Beispiel:")
print(f"Epochs: {slight_overfit['epochs']}")
print(f"Train MSE: {slight_overfit['train_mse']:.3e} | Test MSE: {slight_overfit['test_mse']:.3e}")
print(f"Gap: {slight_overfit['gap']:.3e} (sichtbar, aber Test noch besser als Best-Fit)")

print(f"\n### Clear Overfit Beispiel:")
print(f"Epochs: {clear_overfit['epochs']}")
print(f"Train MSE: {clear_overfit['train_mse']:.3e} | Test MSE: {clear_overfit['test_mse']:.3e}")
print(f"Gap: {clear_overfit['gap']:.3e} (deutlich, Test verschlechtert sich)")
