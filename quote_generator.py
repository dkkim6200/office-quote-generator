from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf

import model_builder

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
tf.train.latest_checkpoint(checkpoint_dir)

input_text = open('office_script.txt', 'rb').read().decode(encoding ='utf-8')

print ('Length of text: {} characters'.format(len(input_text)))
vocab = sorted(set(input_text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

model = model_builder.build_model(vocab_size=len(vocab), embedding_dim=256, rnn_units=1024, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 10000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


print("\n\n\n======================= GENERATED TEXT =====================\n\n")
print(generate_text(model, start_string=u"""
Season 10 - Episode 1

                "Peter Ewart"
                
                Written by Daekun Kim
                
                Directed by Shailee Shah
                
                Original Air Date: August 30th, 2019
                
"""))
print("\n\n\n")
