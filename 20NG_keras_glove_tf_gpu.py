import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pathlib
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers

'''
Downloading the 20NG files
'''

# data_path = keras.utils.get_file(
#     "news20.tar.gz",
#     "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz",
#     untar=True,
# )
# print(data_path)
# data_dir = pathlib.Path(data_path).parent / "20_newsgroup"
data_dir = "C:/Users/Admin/.keras/datasets/20_newsgroup"
# print(data_dir)
dirnames = os.listdir(data_dir)
print("Number of directories:", len(dirnames))
print("Directory names:", dirnames)

fnames = os.listdir("C:/Users/Admin/.keras/datasets/20_newsgroup/comp.graphics")
print("Number of files in comp.graphics:", len(fnames))
print("Some example filenames:", fnames[:5])

# print(open("C:/Users/Admin/.keras/datasets/20_newsgroup/comp.graphics/37261").read())

samples = []
labels = []
class_names = []
class_index = 0
for dirname in sorted(os.listdir(data_dir)):
    class_names.append(dirname)
    # print(dirname)
    # print(data_dir)
    dirpath = data_dir + "/" + dirname
    fnames = os.listdir(dirpath)
    print("Processing %s, %d files found" % (dirname, len(fnames)))
    for fname in fnames:
        fpath = dirpath + "/" + fname
        f = open(fpath, encoding="latin-1")
        content = f.read()
        # print("content")
        # print(content)
        # print("\n")
        lines = content.split("\n")
        # print("lines")
        # print(lines)
        # print("\n")
        lines = lines[10:]
        # print("lines")
        # print(lines)
        # print("\n")
        content = "\n".join(lines)
        # print("content")
        # print(content)
        # print("\n")
        samples.append(content)
        labels.append(class_index)
        # break
    class_index += 1
    # break

print("Classes:", class_names)
print("Number of samples:", len(samples))

# Shuffle the data
seed = 1337
rng = np.random.RandomState(seed)
rng.shuffle(samples)
rng = np.random.RandomState(seed)
rng.shuffle(labels)

# Extract a training & validation split
validation_split = 0.2
num_validation_samples = int(validation_split * len(samples))
print(num_validation_samples)
train_samples = samples[:-num_validation_samples]
val_samples = samples[-num_validation_samples:]
train_labels = labels[:-num_validation_samples]
val_labels = labels[-num_validation_samples:]

"""
## Create a vocabulary index
We will be using `TextVectorization` to index the vocabulary found in the dataset.
Later, we'll use the same layer instance to vectorize the samples.
Our layer will only consider the top 30,000 words, and will truncate or pad sequences to
be actually 200 tokens long.
"""

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

vectorizer = TextVectorization(max_tokens=30000, output_sequence_length=300)
text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(128)
vectorizer.adapt(text_ds)

# print(vectorizer.get_vocabulary()[:5])
#
# output = vectorizer([["the cat sat on the mat"]])
# print(output.numpy()[0, :6])

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

path_to_glove_file = "D:/Text Classification Data Disca/glove.6B/glove.6B.100d.txt"
# path_to_glove_file = os.path.join(
#     os.path.expanduser("~"), ".keras/datasets/glove.6B.100d.txt"
# )

embeddings_index = {}
with open(path_to_glove_file, encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        # print(word)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    # print(word)
    # print(embedding_vector)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1

print("Converted %d words (%d misses)" % (hits, misses))

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False, # since we do not want to update them during training
)



int_sequences_input = keras.Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(int_sequences_input, preds)
print(model.summary())

x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()

y_train = np.array(train_labels)
y_val = np.array(val_labels)

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["acc"]
)
model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_val, y_val))

# End-to-End Model
string_input = keras.Input(shape=(1,), dtype="string")
x = vectorizer(string_input)
preds = model(x)
end_to_end_model = keras.Model(string_input, preds)

probabilities = end_to_end_model.predict(
    [["this message is about computer graphics and 3D modeling"]]
)

print(class_names[np.argmax(probabilities[0])])