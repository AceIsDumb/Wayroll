from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from keras.layers import TFSMLayer
from keras.models import Sequential

app = Flask(__name__)
CORS(app)

model = None
tfsm_layer_instance = None

try:
    tfsm_layer_instance = TFSMLayer("model.savedmodel", call_endpoint="serving_default")
    print("TFSMLayer loaded from SavedModel successfully.")

    model = Sequential()

    model.add(tf.keras.Input(shape=(64, 64, 1)))
    model.add(tfsm_layer_instance)

    print("Sequential model built around TFSMLayer.")

except Exception as e:
    print(f"Failed to load TFSMLayer or build Sequential model: {e}")
    model = None

def geojson2img(geojson_data, imgsize=(64, 64), linecol=(255), linewidth=1):
    width, height = imgsize

    try:
        coordinates = geojson_data['geometry']['coordinates']
        if not coordinates:
            print("No coordinates found in GeoJSON.")
            return np.zeros((height, width, 1), dtype=np.float32)

        all_lon = [point[0] for segment in coordinates for point in segment]
        all_lat = [point[1] for segment in coordinates for point in segment]

        min_lon, max_lon = min(all_lon), max(all_lon)
        min_lat, max_lat = min(all_lat), max(all_lat)

        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat

        if lon_range == 0: lon_range = 0.0001
        if lat_range == 0: lat_range = 0.0001

        img = Image.new('L', (width, height), color=0)
        draw = ImageDraw.Draw(img)

        pixel_coords_segments = []
        for segment in coordinates:
            segment_pixels = []
            for lon, lat in segment:
                x_pixel = int(((lon - min_lon) / lon_range) * (width - 1))
                y_pixel = int((1 - (lat - min_lat) / lat_range) * (height - 1))
                segment_pixels.append((x_pixel, y_pixel))
            pixel_coords_segments.append(segment_pixels)

        for segment_pixels in pixel_coords_segments:
            if len(segment_pixels) > 1:
                draw.line(segment_pixels, fill=linecol, width=linewidth)

        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = img_array / 255.0

        return img_array

    except Exception as e:
        print(f"Error converting GeoJSON to image: {e}")
        return np.zeros((height, width, 1), dtype=np.float32)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Cannot make predictions."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received."}), 400

    if 'geometry' not in data or 'coordinates' not in data.get('geometry', {}) or \
       data.get('geometry', {}).get('type') != 'LineString':
        return jsonify({"error": "Invalid GeoJSON format. Expecting a LineString geometry within 'geometry'."}), 400

    imginput = geojson2img(data, imgsize=(64, 64), linecol=(255), linewidth=2)

    if imginput is None:
        return jsonify({"error": "Failed to convert GeoJSON to image."}), 500

    inputbatch = np.expand_dims(imginput, axis=0)

    try:
        predout = model.predict(inputbatch)

        prediction = predout[0][0]

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"Prediction failed due to model error: {str(e)}"}), 500

    return jsonify({
        "accessible": float(prediction) >= 0.5,
        "score": float(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
