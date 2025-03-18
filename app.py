from flask import Flask, request, send_file, render_template_string, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
import base64

app = Flask(__name__, template_folder='templates')
loaded_model = load_model('crack_detector.keras')

# Функция get_heatmap (без изменений)
def get_heatmap(model, img_array, last_conv_layer_name="conv2d_1", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10
    return heatmap

# Функция overlay_heatmap (без изменений)
def overlay_heatmap(heatmap, img_array, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = tf.keras.preprocessing.image.array_to_img(heatmap[..., np.newaxis], scale=False)
    heatmap = heatmap.resize((120, 120))
    heatmap = img_to_array(heatmap)[..., 0]
    heatmap = np.uint8(255 * heatmap / (np.max(heatmap) + 1e-10))
    jet = plt.get_cmap("jet")
    heatmap_color = jet(np.uint8(heatmap))[..., :3] * 255
    superimposed_img = heatmap_color * alpha + img_array * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

@app.route('/')
def home():
    with open('templates/index.html', 'r') as f:
        return render_template_string(f.read())

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST': #проверяем метод
        if 'image' not in request.files:
            return "No image uploaded", 400
        
        file = request.files['image'] #получаем изображение
        img = Image.open(file.stream).resize((120, 120)).convert('RGB')  # преобразование в RGB
        img_array = np.array(img) / 255.0 #преобразовываем изображение в массив 
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        prediction = loaded_model.predict(img_array_expanded)[0][0] #делаем предикт
        label = "POSITIVE" if prediction > 0.5 else "NEGATIVE"
        
        heatmap = get_heatmap(loaded_model, img_array_expanded) #загружаем тепловую карту
        superimposed_img = overlay_heatmap(heatmap, np.array(img)) #накладываем
        
        img_io = io.BytesIO() 
        Image.fromarray(superimposed_img).save(img_io, 'PNG') #преобразовываем массив в изображение
        img_io.seek(0) #перемещаем указатель в начало потока
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8') #кодируем

        return render_template('predict.html', 
                             result=f"Prediction: {label} (confidence: {prediction:.4f})",
                             heatmap=img_base64)
    else:  # GET-запрос
        pass
    return render_template("predict.html")

if __name__ == '__main__':
    app.run(debug=True)