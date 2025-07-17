from tensorflow.keras.models import load_model

model = load_model("CNN_model.keras", compile=False)
model.save("CNN_model.h5")
