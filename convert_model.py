from tensorflow.keras.models import load_model

# Load your .keras model (saved using TF 2.15+)
model = load_model("CNN_model.keras", compile=False)
model.export("saved_model")  
print("Saved model in 'saved_model/' using Keras 3 export")
