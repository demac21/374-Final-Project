
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1. Load CSV and Process Structured Data
# -------------------------------
# Assuming your CSV is loaded into the DataFrame `df`
df = pd.read_csv('/content/drive/MyDrive/CSC 374/Project/Data/draft_data2.csv')

# Rename column if necessary
df.rename(columns={'metro': 'METRO'}, inplace=True)

# Define the columns for structured features.
numeric_features = ['BEDS', 'BATHS', 'SQUARE FEET', 'LOT SIZE', 'YEAR BUILT', 'DAYS ON MARKET', '$/SQUARE FEET', 'LATITUDE', 'LONGITUDE']
categorical_features = ['PROPERTY TYPE', 'CITY', 'STATE OR PROVINCE']

# Process numeric features: extract and scale them.
scaler = StandardScaler()
df_numeric_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_features]), columns=numeric_features)

# Process categorical features using one-hot encoding.
df_categorical = pd.get_dummies(df[categorical_features], drop_first=True)

# Combine numeric and one-hot encoded categorical features.
df_structured = pd.concat([df_numeric_scaled, df_categorical], axis=1)
X_structured = df_structured.values

# Scale target variable (PRICE) as well.
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(df['PRICE'].values.reshape(-1, 1)).flatten()

# -------------------------------
# 2. Prepare Image File Paths and Build a tf.data Pipeline
# -------------------------------

def get_full_image_path(row):
    metro = row['METRO']
    file_name = row['IMAGE_FILE']
    return f'/content/drive/MyDrive/CSC 374/Project/Data/images/original_images/{metro}_images/{file_name}'

df['full_image_path'] = df.apply(get_full_image_path, axis=1)
image_paths = df['full_image_path'].tolist()

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = preprocess_input(image)
    return image

def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image

AUTOTUNE = tf.data.AUTOTUNE
ds_images = tf.data.Dataset.from_tensor_slices(image_paths)
ds_images = ds_images.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
ds_images = ds_images.map(augment, num_parallel_calls=AUTOTUNE)
ds_images = ds_images.cache('/tmp/dataset_cache')
batch_size = 8
ds_images = ds_images.batch(batch_size).prefetch(AUTOTUNE)

# -------------------------------
# 3. Split Data for Training and Validation
# -------------------------------

# Split structured data and scaled target.
X_struct_train, X_struct_val, y_train, y_val = train_test_split(
    X_structured, y_scaled, test_size=0.2, random_state=42
)

# Also split the image file paths accordingly.
train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

# Build separate datasets for training and validation images.
ds_images_train = tf.data.Dataset.from_tensor_slices(train_paths)
ds_images_train = ds_images_train.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
ds_images_train = ds_images_train.map(augment, num_parallel_calls=AUTOTUNE)
ds_images_train = ds_images_train.batch(batch_size).prefetch(AUTOTUNE)

ds_images_val = tf.data.Dataset.from_tensor_slices(val_paths)
ds_images_val = ds_images_val.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
ds_images_val = ds_images_val.batch(batch_size).prefetch(AUTOTUNE)

# -------------------------------
# 4. Building the Multimodal Model
# -------------------------------

# --- Image Branch ---
image_input = Input(shape=(224, 224, 3), name='image_input')
aug = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1)
], name='image_augmentation')
augmented_images = aug(image_input)

base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=augmented_images)
base_model.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

# --- Structured Data Branch ---
structured_input = Input(shape=(X_struct_train.shape[1],), name='structured_input')
y_branch = Dense(32, activation='relu')(structured_input)
y_branch = Dropout(0.2)(y_branch)

# --- Fusion ---
combined = Concatenate()([x, y_branch])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.3)(z)
output = Dense(1, activation='linear', name='price_output')(z)

model = Model(inputs=[image_input, structured_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
model.summary()

# -------------------------------
# 5. Model Training
# -------------------------------

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('/content/drive/MyDrive/CSC 374/Project/Data/results/best_model2.h5', monitor='val_loss', save_best_only=True)
]

# Create datasets for the structured data.
ds_struct_train = tf.data.Dataset.from_tensor_slices(X_struct_train.astype(np.float32)).batch(batch_size)
ds_struct_val = tf.data.Dataset.from_tensor_slices(X_struct_val.astype(np.float32)).batch(batch_size)

# Zip the image and structured data datasets.
train_dataset = tf.data.Dataset.zip((
    {'image_input': ds_images_train, 'structured_input': ds_struct_train},
    tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size)
))
val_dataset = tf.data.Dataset.zip((
    {'image_input': ds_images_val, 'structured_input': ds_struct_val},
    tf.data.Dataset.from_tensor_slices(y_val).batch(batch_size)
))

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3,
    callbacks=callbacks
)

# -------------------------------
# 6. Model Evaluation and Predictions
# -------------------------------

loss, mae = model.evaluate(val_dataset)
print(f"Validation MSE: {loss:.4f}, MAE: {mae:.4f}")

predictions = model.predict(val_dataset)
print("Sample Predictions:", predictions[:5].flatten())

predictions_original_scale = target_scaler.inverse_transform(predictions.reshape(-1, 1))
print("Sample Predictions (Original Scale):", predictions_original_scale[:5].flatten())

with open("/content/drive/MyDrive/CSC 374/Project/Data/results/results2.txt", "w") as file:
    file.write(f"Validation MSE: {loss:.4f}, MAE: {mae:.4f}\n")
    file.write("Sample Predictions (Original Scale): " + str(predictions_original_scale[:5].flatten()) + "\n")

model.save('/content/drive/MyDrive/CSC 374/Project/Data/models/model2.keras')

# Reshape predictions to 2D before inverse transforming.
predictions_original_scale = target_scaler.inverse_transform(predictions.reshape(-1, 1))
print("Sample Predictions (Original Scale):", predictions_original_scale[:5].flatten())

with open("/content/drive/MyDrive/CSC 374/Project/Data/results/results.txt", "w") as file:
    file.write(f"Validation MSE: {loss:.4f}, MAE: {mae:.4f}\n")
    file.write("Sample Predictions (Original Scale): " + str(predictions_original_scale[:5].flatten()) + "\n")

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assuming y_train and y_test are your training and testing targets in the original (or scaled) space.
dummy = DummyRegressor(strategy="mean")
dummy.fit(np.zeros_like(y_train).reshape(-1, 1), y_train)  # Dummy feature input since only target matters.
y_dummy_pred = dummy.predict(np.zeros_like(y_val).reshape(-1, 1))

mse_dummy = mean_squared_error(y_val, y_dummy_pred)
mae_dummy = mean_absolute_error(y_val, y_dummy_pred)

print(f"Dummy Baseline - MSE: {mse_dummy:.4f}, MAE: {mae_dummy:.4f}")

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_struct_train, y_train)
y_lin_pred = lin_reg.predict(X_struct_val)

mse_lin = mean_squared_error(y_val, y_lin_pred)
mae_lin = mean_absolute_error(y_val, y_lin_pred)

print(f"Linear Regression Baseline - MSE: {mse_lin:.4f}, MAE: {mae_lin:.4f}")

# Write the results to a text file.
with open("/content/drive/MyDrive/CSC 374/Project/Data/results/m1baseline_results.txt", "w") as file:
    file.write("Baseline Results:\n")
    file.write("Dummy Regressor - MSE: {:.4f}, MAE: {:.4f}\n".format(mse_dummy, mae_dummy))
    file.write("Linear Regression - MSE: {:.4f}, MAE: {:.4f}\n".format(mse_lin, mae_lin))