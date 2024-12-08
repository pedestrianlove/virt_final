import tensorflow as tf
import time
import argparse
import os
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description="TensorFlow Multi-GPU Training")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for training")
args = parser.parse_args()

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
if args.gpus > len(gpus):
    raise RuntimeError(f"Requested {args.gpus} GPUs, but only {len(gpus)} are available.")

# Set GPUs to visible for TensorFlow
tf.config.experimental.set_visible_devices(gpus[:args.gpus], 'GPU')
for gpu in gpus[:args.gpus]:
    tf.config.experimental.set_memory_growth(gpu, False)

# Define a simple model and dataset
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),  # Flatten the input to match Dense layers
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def create_large_dataset(BATCH_MULTIPLIER = 1):
    # Generate a synthetic large dataset
    num_samples = 100000  # Increase the number of samples to make the dataset larger
    x_train = np.random.rand(num_samples, 32, 32, 3).astype('float32')
    y_train = np.random.randint(0, 10, size=(num_samples,))
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000).batch(10240 * BATCH_MULTIPLIER).prefetch(tf.data.AUTOTUNE)
    return dataset

# Train the model and measure performance
def train_model(model, dataset, epochs=5):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    start_time = time.time()
    model.fit(dataset, epochs=epochs, verbose=0)
    end_time = time.time()
    return end_time - start_time

# Multi-GPU Training using Strategy
def multi_gpu_training(dataset):
    print(f"Training on {args.gpus} GPUs using tf.distribute.Strategy...")
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {strategy.num_replicas_in_sync} devices for training.")

    with strategy.scope():
        model = create_model()
        duration = train_model(model, dataset)
    print(f"Multi-GPU training completed in {duration:.2f} seconds.")
    return duration

# Main logic
if args.gpus == 1:
    print("Generating dataset...")
    dataset = create_large_dataset(1)
    print("Training on a single GPU...")
    with tf.device('/GPU:0'):
        model = create_model()
        duration = train_model(model, dataset)
    print(f"Single GPU training completed in {duration:.2f} seconds.")
elif args.gpus > 1:
    print("Generating dataset...")
    dataset = create_large_dataset(args.gpus)
    multi_gpu_time = multi_gpu_training(dataset)
    print(f"{args.gpus} GPUs Training Time: {multi_gpu_time:.2f} seconds.")
else:
    print("Invalid number of GPUs specified.")
