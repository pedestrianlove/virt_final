import optree
import tensorflow as tf
import time
import argparse
import os
import numpy as np
import tensorflow_datasets as tfds

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
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model

def create_large_dataset(BATCH_MULTIPLIER = 1):
    def scale(image, label):
      image = tf.cast(image, tf.float32)
      image /= 255

      return image, label

    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']
    train = mnist_train.map(scale).cache().shuffle(10000).batch(64 * BATCH_MULTIPLIER)
    test = mnist_test.map(scale).batch(1024 * BATCH_MULTIPLIER)
    return train, test

# Train the model and measure performance
def train_model(model, dataset, epochs=5):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
    start_time = time.time()
    model.fit(dataset, epochs=epochs, verbose=1)
    end_time = time.time()
    return end_time - start_time

# Multi-GPU Training using Strategy
def multi_gpu_training(dataset, test_data):
    print(f"Training on {args.gpus} GPUs using tf.distribute.Strategy...")
    # communication_options = tf.distribute.experimental.CommunicationOptions(
    #     implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
    # strategy = tf.distribute.MultiWorkerMirroredStrategy(
    #     communication_options=communication_options)
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {strategy.num_replicas_in_sync} devices for training.")

    with strategy.scope():
        model = create_model()
        duration = train_model(model, dataset)
    print(f"{args.gpus} GPUs Training Time: {duration:.2f} seconds.")

    # Validation
    eval_loss, eval_acc = model.evaluate(test_data)
    print('Eval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
    return duration

# Main logic
if args.gpus == 1:
    print("Generating dataset...")
    train_data, test_data = create_large_dataset(1)
    print("Training on a single GPU...")
    with tf.device('/GPU:0'):
        model = create_model()
        duration = train_model(model, train_data)
    print(f"Single GPU training completed in {duration:.2f} seconds.")

    # Validation
    eval_loss, eval_acc = model.evaluate(test_data)
    print('Eval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
elif args.gpus > 1:
    print("Generating dataset...")
    train_data, test_data = create_large_dataset(args.gpus)
    multi_gpu_time = multi_gpu_training(train_data, test_data)

else:
    print("Invalid number of GPUs specified.")

