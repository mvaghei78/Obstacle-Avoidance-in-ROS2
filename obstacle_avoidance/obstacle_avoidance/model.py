from tensorflow.keras.models import model_from_json
def load_dronet(model_path, weights_path):
    # Load the model architecture from JSON

    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # Load weights into the model

    loaded_model.load_weights(weights_path)
    # Compile the model (if needed)
    # Replace optimizer, loss, and metrics with the ones you used during training
    loaded_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return loaded_model

if __name__ == '__main__':
    model_path = './dronet/model_struct.json'
    weights_path = './dronet/best_weights.h5'
    load_dronet(model_path, weights_path)