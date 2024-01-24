import os
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, concatenate
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Update the base path to the folder location on Google Drive
google_drive_base_path = '/content/drive/My Drive/Colab Notebooks/YearlingPictures/'

# Correct the path to the CSV file
csv_file_path = os.path.join(google_drive_base_path, 'horse_data.csv')

# Load the data
data = pd.read_csv(csv_file_path)

# Preprocess the images
def preprocess_image(image_url):
    image_filename = os.path.basename(image_url)
    image_path = os.path.join(google_drive_base_path, image_filename)
    if not os.path.exists(image_path):
        return np.nan
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    return image

# Apply the preprocessing function to the image URLs in the DataFrame
data['processed_image'] = data['image_url'].apply(preprocess_image)

# Drop rows with missing images or price
data.dropna(subset=['processed_image', 'price'], inplace=True)

# One-hot encode the categorical features
categorical_features = ['color', 'sex', 'sire', 'dam', 'damsire', 'bonus_scheme']
one_hot_encoder = OneHotEncoder(sparse=False)
encoded_categorical = one_hot_encoder.fit_transform(data[categorical_features])

# Normalize the lot_number column
scaler = StandardScaler()
data['lot_number'] = scaler.fit_transform(data[['lot_number']].astype(float))

# Prepare the target variable (price)
data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)

# Split into features and target
X_images = np.stack(data['processed_image'].values)
X_categorical = encoded_categorical
X_numerical = data[['lot_number']].values
y = data['price'].values

# Split the data into training and test sets
X_train_images, X_test_images, X_train_categorical, X_test_categorical, X_train_numerical, X_test_numerical, y_train, y_test = train_test_split(
    X_images, X_categorical, X_numerical, y, test_size=0.2, random_state=42
)

# Load a pre-trained ResNet50 model for image processing
base_model = ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False

# Image processing branch
image_input = Input(shape=(224, 224, 3))
x_image = base_model(image_input)
x_image = Flatten()(x_image)

# Categorical features branch
categorical_input = Input(shape=(X_train_categorical.shape[1],))
x_categorical = Dense(64, activation='relu')(categorical_input)

# Numerical features branch
numerical_input = Input(shape=(1,))
x_numerical = Dense(64, activation='relu')(numerical_input)

# Combine branches
combined = concatenate([x_image, x_categorical, x_numerical])
combined = Dense(64, activation='relu')(combined)
output = Dense(1)(combined)

model = Model(inputs=[image_input, categorical_input, numerical_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(
    [X_train_images, X_train_categorical, X_train_numerical],
    y_train,
    validation_split=0.1,
    epochs=10
)

# Function to evaluate the model
def evaluate_model(model, X_test_images, X_test_categorical, X_test_numerical, y_test):
    # Predict on the test data
    test_predictions = model.predict([X_test_images, X_test_categorical, X_test_numerical])
    
    # Calculate MSE and RMSE
    mse = mean_squared_error(y_test, test_predictions)
    rmse = math.sqrt(mse)

    return mse, rmse

# Evaluate the model
test_mse, test_rmse = evaluate_model(model, X_test_images, X_test_categorical, X_test_numerical, y_test)
print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)
