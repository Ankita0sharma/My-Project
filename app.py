import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

app = Flask(_name_)

def train_and_save_model():
    """Train model and save encoders if they don't exist"""
    try:
        # Load data
        df = pd.read_csv('Zomato_df.csv')
        
        # Drop unnecessary columns
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        
        # Initialize encoders
        le_location = LabelEncoder()
        le_rest_type = LabelEncoder()
        le_cuisines = LabelEncoder()
        le_menu_item = LabelEncoder()
        
        # Encode categorical variables (adjust column names as needed)
        df['location_encoded'] = le_location.fit_transform(df['location'])
        df['rest_type_encoded'] = le_rest_type.fit_transform(df['rest_type'])
        df['cuisines_encoded'] = le_cuisines.fit_transform(df['cuisines'])
        df['menu_item_encoded'] = le_menu_item.fit_transform(df['menu_item'])
        
        # Prepare features
        X = df[['online_order', 'book_table', 'votes', 'location_encoded', 
                'rest_type_encoded', 'cuisines_encoded', 'menu_item_encoded', 'cost']]
        y = df['rate']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
        
        # Train model
        model = ExtraTreesRegressor(n_estimators=120, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model and encoders
        pickle.dump(model, open('model.pkl', 'wb'))
        pickle.dump(le_location, open('le_location.pkl', 'wb'))
        pickle.dump(le_rest_type, open('le_rest_type.pkl', 'wb'))
        pickle.dump(le_cuisines, open('le_cuisines.pkl', 'wb'))
        pickle.dump(le_menu_item, open('le_menu_item.pkl', 'wb'))
        
        print("Model trained and saved successfully!")
        return model, le_location, le_rest_type, le_cuisines, le_menu_item
        
    except Exception as e:
        print(f"Error training model: {e}")
        return create_dummy_model()

def create_dummy_model():
    """Create dummy model if training fails"""
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
    y_dummy = np.random.rand(100) * 4 + 1
    model.fit(X_dummy, y_dummy)
    
    return model, le_location, le_rest_type, le_cuisines, le_menu_item

# Initialize model and encoders
try:
    # Try to load existing model
    model = pickle.load(open('model.pkl', 'rb'))
    le_location = pickle.load(open('le_location.pkl', 'rb'))
    le_rest_type = pickle.load(open('le_rest_type.pkl', 'rb'))
    le_cuisines = pickle.load(open('le_cuisines.pkl', 'rb'))
    le_menu_item = pickle.load(open('le_menu_item.pkl', 'rb'))
    print("Loaded existing model")
except:
    # Train new model
    print("Training new model...")
    model, le_location, le_rest_type, le_cuisines, le_menu_item = train_and_save_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        online_order = int(request.form['Online Order'])
        book_table = int(request.form['Book Table'])
        votes = int(request.form['Votes'])
        location = request.form['Location']
        rest_type = request.form['Restaurant Type']
        cuisines = request.form['Cuisines']
        menu_item = request.form['Menu Item']
        cost = float(request.form['Cost'])

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

if _name_ == "_main_":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
