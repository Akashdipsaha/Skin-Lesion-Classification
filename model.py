import tensorflow as tf

class Model:
    def __init__(self, model_path):
        # Load the pre-trained model
        self.model = tf.keras.models.load_model(model_path)
        # Class labels for the 8 skin disease categories
        self.class_names = ['Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis',
                            'Dermatofibroma', 'Melanocytic nevus', 'Melanoma',
                            'Squamous cell carcinoma', 'Vascular lesion']

    def preprocess_image(self, image):
        # Resize the image to the expected input shape for EfficientNetB1 (240x240)
        image = tf.image.resize(image, (240, 240))
        # Normalize pixel values to the range [0, 1]
        image = image / 255.0
        # Expand dimensions to create a batch size of 1
        image = tf.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        # Preprocess the input image
        image = self.preprocess_image(image)
        # Predict using the loaded model
        predictions = self.model.predict(image)
        # Get the predicted class index with the highest probability
        predicted_class = tf.argmax(predictions[0])
        # Return the predicted class name and confidence score
        predicted_label = self.class_names[predicted_class]
        confidence = tf.reduce_max(predictions[0]) * 100
        return predicted_label, confidence.numpy()
