from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Input(shape=(224, 224, 3), name="input_layer"),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(2, activation="softmax")
])

# Save in .keras format for Streamlit compatibility
model.save("CNN_model.keras")
print("Saved model in 'CNN_model.keras' format.")