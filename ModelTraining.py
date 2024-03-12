import numpy as np
import nnfs
from ModelClasses import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

nnfs.init()

labels = os.listdir('PokemonData')
# X, y= create_data_mnist('PokemonData')
X, y = input_target_split('PokemonData', labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)
X_train_reshaped = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test_reshaped = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test),
            epochs=10, batch_size=128, print_every=100)

model.save('pokemon.model')