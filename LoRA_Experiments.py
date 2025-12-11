"""LoRA_Experiments.py

TensorFlow-based LoRA experiment sketch for translation. Default is dry-run to avoid
downloads and long training. Pass --run to execute the heavy flow (requires TF, TFA,
transformers, datasets, GPU/TPU recommended).
"""

import argparse
import logging
from typing import Tuple


def count_params_tf(model) -> Tuple[int, int]:
    import numpy as np
    trainable_params = int(np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights]))
    non_trainable_params = int(np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights]))
    return trainable_params, non_trainable_params


def run_experiment(max_train: int = 20000, max_eval: int = 1000, ranks=None, batch_sizes=None, epochs: int = 2):
    import tensorflow as tf
    import tensorflow_addons as tfa  # noqa: F401 - included for compatibility if you add optimizers
    import tf_keras
    from datasets import load_dataset
    from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

    ranks = ranks or [1, 4, 16]
    batch_sizes = batch_sizes or [8, 64, 128]
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

    def preprocess_data(examples):
        inputs = [f'Translate English to German: {example["en"]}' for example in examples['translation']]
        targets = [example['de'] for example in examples['translation']]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length', return_tensors='tf')
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length', return_tensors='tf').input_ids
        model_inputs['labels'] = labels
        decoder_inputs = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["decoder_input_ids"] = decoder_inputs["input_ids"]
        return model_inputs

    class LoRALayer(tf.keras.layers.Layer):
        def __init__(self, dense, rank=4):
            super().__init__()
            self.dense = dense
            self.rank = rank

        def build(self, input_shape):
            self.w_a = self.add_weight(shape=(input_shape[-1], self.rank), initializer='random_normal', trainable=True, name='w_a')
            self.w_b = self.add_weight(shape=(self.rank, self.dense.units), initializer='random_normal', trainable=True, name='w_b')

        def call(self, inputs):
            original_output = self.dense(inputs)
            lora_output = tf.matmul(tf.matmul(inputs, self.w_a), self.w_b)
            self.dense.trainable = False
            return original_output + lora_output

    log = logging.getLogger("lora_experiments")
    log.info("Loading dataset wmt16 de-en (train subset %s, eval subset %s)...", max_train, max_eval)
    dataset = load_dataset('wmt16', 'de-en')
    train_dataset = dataset['train'].select(range(max_train)).map(preprocess_data, batched=True)
    test_dataset = dataset['test'].select(range(max_eval)).map(preprocess_data, batched=True)

    results = {}
    for rank in ranks:
        for batch_size in batch_sizes:
            log.info("Training with rank=%s, batch_size=%s", rank, batch_size)
            model = TFAutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
            for layer in model.layers[:3]:
                layer.trainable = False
            model.layers[3] = LoRALayer(model.get_layer('lm_head'), rank=rank)

            trainable_params, non_trainable_params = count_params_tf(model)
            log.info("Params | trainable: %d | non-trainable: %d", trainable_params, non_trainable_params)

            train_tf = train_dataset.to_tf_dataset(
                columns=['input_ids', 'attention_mask', 'decoder_input_ids'],
                label_cols=['labels'],
                shuffle=True,
                batch_size=batch_size,
                collate_fn=None,
            )

            eval_tf = test_dataset.to_tf_dataset(
                columns=['input_ids', 'attention_mask', 'decoder_input_ids'],
                label_cols=['labels'],
                shuffle=False,
                batch_size=batch_size,
                collate_fn=None,
            )

            model.compile(
                optimizer=tf_keras.optimizers.Adam(learning_rate=1e-2),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            )

            history = model.fit(train_tf, validation_data=eval_tf, epochs=epochs)
            results[(rank, batch_size)] = history.history

    for (rank, batch_size), history in results.items():
        log.info("Results for rank=%s, batch_size=%s: %s", rank, batch_size, history)


def main():
    parser = argparse.ArgumentParser(description="Heavy TensorFlow LoRA experiment. Defaults to dry-run to avoid downloads.")
    parser.add_argument("--run", action="store_true", help="Actually run the experiment (downloads data/models; heavy).")
    parser.add_argument("--max_train", type=int, default=2000, help="Train subset size for quick runs")
    parser.add_argument("--max_eval", type=int, default=200, help="Eval subset size for quick runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("lora_experiments")

    if not args.run:
        log.info("[DRY RUN] Skipping heavy TensorFlow LoRA experiment.")
        log.info("Pass --run to download wmt16 de-en, load flan-t5-base, and train LoRA-wrapped head.")
        return 0

    run_experiment(max_train=args.max_train, max_eval=args.max_eval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
!pip install transformers tensorflow datasets tensorflow_addons

from datasets import load_dataset

# Load the WMT16 English-German dataset
dataset = load_dataset('wmt16', 'de-en')

# Display an example
print(dataset['train'][0])

import tensorflow as tf
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

# Preprocess the dataset for input into the model
def preprocess_data(examples):
    inputs = [f'Translate English to German: {example["en"]}' for example in examples['translation']]
    targets = [example['de'] for example in examples['translation']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length', return_tensors='tf')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length', return_tensors='tf').input_ids
    model_inputs['labels'] = labels
    decoder_inputs = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["decoder_input_ids"] = decoder_inputs["input_ids"]
    return model_inputs





# Replace the dense layers with LoRA layers
class LoRALayer(tf.keras.layers.Layer):
    def __init__(self, dense, rank=4):
        super().__init__()
        self.dense = dense
        self.rank = rank

    def build(self, input_shape):
        self.w_a = self.add_weight(shape=(input_shape[-1], self.rank),
                                   initializer='random_normal',
                                   trainable=True, name='w_a')
        self.w_b = self.add_weight(shape=(self.rank, self.dense.units),
                                   initializer='random_normal',
                                   trainable=True, name='w_b')

    def call(self, inputs):
        original_output = self.dense(inputs)
        lora_output = tf.matmul(tf.matmul(inputs, self.w_a), self.w_b)
        self.dense.trainable = False
        return original_output + lora_output




import tf_keras
import numpy as np
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense


def count_params(model):
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
    return trainable_params, non_trainable_params

# Define training configurations
ranks = [1, 4, 16]
batch_sizes = [8, 64, 128]
epochs = 2
results = {}

for rank in ranks:
    for batch_size in batch_sizes:
        print(f"Training with rank={rank}, batch_size={batch_size}")
        model = TFAutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
        model.layers[0].trainable = False
        model.layers[1].trainable = False
        model.layers[2].trainable = False
        model.layers[3] = LoRALayer(model.get_layer('lm_head'))

        # Get the number of parameters
        trainable_params, non_trainable_params = count_params(model)

        # Print the number of parameters
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {non_trainable_params}")

        # Update the batch size

        train_dataset = dataset['train'].select(range(20000)).map(preprocess_data, batched=True)
        test_dataset = dataset['test'].select(range(1000)).map(preprocess_data, batched=True)

        train_dataset =  train_dataset.to_tf_dataset(
            columns=['input_ids', 'attention_mask', 'decoder_input_ids'],
            label_cols=['labels'],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=None
        )

        test_dataset = test_dataset.to_tf_dataset(
            columns=['input_ids', 'attention_mask', 'decoder_input_ids'],
            label_cols=['labels'],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=None
        )

        # Compile the model
        model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=1e-2),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        # Train the model
        history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)
        results[(rank, batch_size)] = history.history


# Evaluate the model for each configuration
for (rank, batch_size), history in results.items():
    print(f"Results for rank={rank}, batch_size={batch_size}")
    print(history)




