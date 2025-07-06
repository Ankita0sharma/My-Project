import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor

app = Flask(_name_)

def create_dummy_model_and_encoders():
    """Create dummy model and encoders if files don't exist"""
    # Create dummy label encoders
    le_location = LabelEncoder()
    le_rest_type = LabelEncoder()
    le_cuisines = LabelEncoder()
    le_menu_item = LabelEncoder()
    
    # Fit with dummy data
    le_location.fit(['BTM', 'Koramangala', 'Indiranagar', 'Whitefield', 'Jayanagar'])
    le_rest_type.fit(['Casual Dining', 'Quick Bites', 'Cafe', 'Fine Dining', 'Buffet'])
    le_cuisines.fit(['North Indian', 'South Indian', 'Chinese', 'Italian', 'Continental'])
    le_menu_item.fit(['Biryani', 'Pizza', 'Burger', 'Dosa', 'Noodles'])
    
    # Create dummy model
    model = ExtraTreesRegressor(n_estimators=10, random_state=42)
    X_dummy = np.random.rand(100, 8)
    y_dummy = np.random.rand(100) * 4 + 1  # ratings between 1-5
    model.fit(X_dummy, y_dummy)
    
    return model, le_location, le_rest_type, le_cuisines, le_menu_item

try:
    # Try to load existing model and encoders
    if all(os.path.exists(f) for f in ['model.pkl', 'le_location.pkl', 'le_rest_type.pkl', 'le_cuisines.pkl', 'le_menu_item.pkl']):
        model = pickle.load(open('model.pkl', 'rb'))
        le_location = pickle.load(open('le_location.pkl', 'rb'))
        le_rest_type = pickle.load(open('le_rest_type.pkl', 'rb'))
        le_cuisines = pickle.load(open('le_cuisines.pkl', 'rb'))
        le_menu_item = pickle.load(open('le_menu_item.pkl', 'rb'))
        print("Loaded existing model and encoders")
    else:
        # Create dummy model and encoders
        model, le_location, le_rest_type, le_cuisines, le_menu_item = create_dummy_model_and_encoders()
        print("Created dummy model and encoders")
        
except Exception as e:
    print(f"Error loading model: {e}")
    model, le_location, le_rest_type, le_cuisines, le_menu_item = create_dummy_model_and_encoders()
    print("Created dummy model and encoders due to error")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        online_order = int(request.form.get('Online Order', 0))
        book_table = int(request.form.get('Book Table', 0))
        votes = int(request.form.get('Votes', 0))
        location = request.form.get('Location', 'BTM')
        rest_type = request.form.get('Restaurant Type', 'Casual Dining')
        cuisines = request.form.get('Cuisines', 'North Indian')
        menu_item = request.form.get('Menu Item', 'Biryani')
        cost = float(request.form.get('Cost', 500))

        # Handle unknown labels with fallback
        try:
            location_enc = le_location.transform([location])[0]
        except:
            location_enc = 0

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

        # Create feature array
        input_features = np.array([[online_order, book_table, votes,
                                    location_enc, rest_type_enc, cuisines_enc,
                                    menu_item_enc, cost]])

        # Predict
        prediction = model.predict(input_features)
        output = round(prediction[0], 1)

        return render_template('index.html', prediction_text=f'Predicted Rating: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

@app.route('/health')
def health():
    return {'status': 'healthy'}

if _name_ == "_main_":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
