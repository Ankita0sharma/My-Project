import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model and encoders
model = pickle.load(open('model.pkl', 'rb'))
le_location = pickle.load(open('le_location.pkl', 'rb'))
le_rest_type = pickle.load(open('le_rest_type.pkl', 'rb'))
le_cuisines = pickle.load(open('le_cuisines.pkl', 'rb'))
le_menu_item = pickle.load(open('le_menu_item.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        online_order = int(request.form['Online Order'])
        book_table = int(request.form['Book Table'])
        Votes = int(request.form['Votes'])
        location = request.form['Location']
        rest_type = request.form['Restaurant Type']
        cuisines = request.form['Cuisines']
        menu_item = request.form['Menu Item']
        cost = float(request.form['Cost'])

        # Handle unknown labels with fallback
        try:
            location_enc = le_location.transform([location])[0]
        except:
            location_enc = 0  # fallback or set -1

        try:
            rest_type_enc = le_rest_type.transform([rest_type])[0]
        except:
            rest_type_enc = 0

        try:
            cuisines_enc = le_cuisines.transform([cuisines])[0]
        except:
            cuisines_enc = 0

        try:
            menu_item_enc = le_menu_item.transform([menu_item])[0]
        except:
            menu_item_enc = 0

        # Create feature array (reshape to 2D)
        input_features = np.array([[online_order, book_table, Votes,
                                    location_enc, rest_type_enc, cuisines_enc,
                                    menu_item_enc, cost]])

        # Predict
        prediction = model.predict(input_features)
        output = round(prediction[0], 1)

        return render_template('index.html', prediction_text=f'Predicted Rating: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
