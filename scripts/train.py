import os
import numpy as np
import cv2 # OpenCV for image loading
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam # Or AdamW for potentially better generalization

# ────────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ────────────────────────────────────────────────────────────────────────────────
def create_trainable_attention_model(input_size=224, learning_rate=1e-4):
    """
    Creates a new attention detection model for training.
    Uses MobileNetV2 as a base, pre-trained on ImageNet.
    """
    input_tensor = Input(shape=(input_size, input_size, 3))
    
    # Base model: MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
    
    # Fine-tuning strategy:
    # Option 1: Train all layers of MobileNetV2 (can be slow, requires more data)
    base_model.trainable = True 
    
    # Option 2: Freeze the base model initially (feature extraction)
    # base_model.trainable = False 

    # Option 3: Unfreeze some of the later layers for fine-tuning
    # print(f"Total layers in base_model: {len(base_model.layers)}")
    # fine_tune_at_layer_name = 'block_13_expand' # Example layer name
    # fine_tune_at_index = None
    # for i, layer in enumerate(base_model.layers):
    #     if layer.name == fine_tune_at_layer_name:
    #         fine_tune_at_index = i
    #         break
    # if fine_tune_at_index is not None:
    #     print(f"Fine-tuning from layer: {fine_tune_at_layer_name} (index {fine_tune_at_index})")
    #     for layer in base_model.layers[:fine_tune_at_index]:
    #         layer.trainable = False
    # else:
    #     print(f"Warning: Layer {fine_tune_at_layer_name} not found for setting fine_tune_at_index. Training all layers.")
    #     base_model.trainable = True


    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Regularization
    # Output layer: Dense with 1 unit and sigmoid activation for binary classification
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\HP\Desktop\student_attention_detector\Student-engagement-dataset"
INPUT_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  # Start with this, EarlyStopping will find the best epoch
LEARNING_RATE = 1e-4 # Initial learning rate (can be reduced by ReduceLROnPlateau)
# Note on class labels based on your folder names "Engaged", "Not engaged":
# The script will likely create the following mapping due to alphabetical sorting:
# 'Engaged': 0
# 'Not engaged': 1
# This means your trained model will output ~0 for 'Engaged' and ~1 for 'Not engaged'.
# In your inference.py, you'll then set:
# MODEL_PREDICTS_HIGH_FOR_ATTENTIVE = False (since low score ~0 means 'Engaged')
# ATTENTION_THRESHOLD = 0.5 (or similar, to separate 0 and 1)

# ────────────────────────────────────────────────────────────────────────────────
# DATA LOADING AND PREPROCESSING
# ────────────────────────────────────────────────────────────────────────────────
def load_image_paths_and_labels(dataset_dir):
    image_paths = []
    labels = []
    # Sort class names to ensure consistent mapping
    class_names = sorted(os.listdir(dataset_dir)) 
    
    # Create a label map (e.g., {'Engaged': 0, 'Not engaged': 1})
    label_map = {name: i for i, name in enumerate(class_names)}
    print(f"Dataset classes found: {class_names}")
    print(f"Label map created: {label_map}")
    print("This means the model will learn:")
    for class_name, label_idx in label_map.items():
        print(f"  - Output ~{label_idx} for class '{class_name}'")

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, fname))
                    labels.append(label_map[class_name])
                else:
                    print(f"Skipping non-image file: {os.path.join(class_dir, fname)}")
        else:
            print(f"Warning: Expected directory, but found file: {class_dir}")
            
    return image_paths, np.array(labels)

print(f"Loading dataset from: {DATASET_PATH}")
image_paths, labels = load_image_paths_and_labels(DATASET_PATH)

if not image_paths:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR: No images found in {DATASET_PATH} or its subdirectories ('Engaged', 'Not engaged').")
    print(f"Please check:")
    print(f"1. The DATASET_PATH is correct: {DATASET_PATH}")
    print(f"2. The subfolders 'Engaged' and 'Not engaged' exist directly under DATASET_PATH.")
    print(f"3. The subfolders contain .jpg, .jpeg, or .png image files.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    exit()

print(f"Found {len(image_paths)} images belonging to {len(np.unique(labels))} classes.")
if len(np.unique(labels)) < 2:
    print("ERROR: The dataset must contain at least two classes for training.")
    exit()

# Split Data (Train/Validation)
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels # stratify for balanced splits
)
print(f"Training samples: {len(X_train_paths)}, Validation samples: {len(X_val_paths)}")

# Image Preprocessing and Data Augmentation
def preprocess_image_for_generator(image_path, target_size=(INPUT_SIZE, INPUT_SIZE)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # MobileNetV2 expects RGB
        img = cv2.resize(img, target_size)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
    return img

# Custom data generator
def data_generator(image_paths_list, labels_list, batch_size_val, target_size=(INPUT_SIZE, INPUT_SIZE), augment=False):
    num_samples = len(image_paths_list)
    
    # Define ImageDataGenerator for augmentation or just rescaling
    if augment:
        image_data_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else: # For validation data, only rescale
        image_data_gen = ImageDataGenerator(rescale=1./255)

    while True: # Loop forever so the generator never terminates
        indices = np.arange(num_samples)
        np.random.shuffle(indices) # Shuffle at the start of each epoch
        
        shuffled_image_paths = [image_paths_list[i] for i in indices]
        shuffled_labels = labels_list[indices]

        for offset in range(0, num_samples, batch_size_val):
            batch_img_paths = shuffled_image_paths[offset:offset + batch_size_val]
            batch_lbls = shuffled_labels[offset:offset + batch_size_val]
            
            batch_images_data = []
            valid_batch_labels = []

            for img_path, label_val in zip(batch_img_paths, batch_lbls):
                img = preprocess_image_for_generator(img_path, target_size)
                if img is not None:
                    batch_images_data.append(img)
                    valid_batch_labels.append(label_val)
            
            if not batch_images_data: # If all images in batch failed to load
                continue 

            batch_images_np = np.array(batch_images_data, dtype=np.float32)
            
            # Apply transformations using the .flow() method from ImageDataGenerator
            # It expects X (images) and y (labels)
            # We will take the first (and only) batch from the flow generator
            # Note: .flow() shuffles by default if not specified, but we already shuffled.
            # Pass labels as well if your loss function or metrics need them in a specific format (e.g. one-hot)
            # For binary_crossentropy with sigmoid, simple 0/1 labels are fine.
            
            # The `flow` method itself is a generator. We need to get a batch from it.
            # It's easier to apply random_transform if we are managing batches ourselves.
            if augment:
                transformed_images_list = []
                for i in range(batch_images_np.shape[0]):
                    # random_transform needs a single image
                    transformed_img = image_data_gen.random_transform(batch_images_np[i])
                    transformed_images_list.append(transformed_img)
                final_batch_images = np.array(transformed_images_list)
                # Rescale after augmentation if not done by random_transform
                if image_data_gen.rescale is None: # Check if rescale was part of ImageDataGenerator already
                     final_batch_images = final_batch_images * (1./255) # Manual rescale if needed
            else: # Validation - only rescale
                final_batch_images = image_data_gen.standardize(batch_images_np.astype(np.float32))


            yield final_batch_images, np.array(valid_batch_labels)


train_gen = data_generator(X_train_paths, y_train, BATCH_SIZE, target_size=(INPUT_SIZE, INPUT_SIZE), augment=True)
val_gen = data_generator(X_val_paths, y_val, BATCH_SIZE, target_size=(INPUT_SIZE, INPUT_SIZE), augment=False)

# ────────────────────────────────────────────────────────────────────────────────
# MODEL CREATION AND TRAINING
# ────────────────────────────────────────────────────────────────────────────────
print("Creating model...")
model = create_trainable_attention_model(input_size=INPUT_SIZE, learning_rate=LEARNING_RATE)
model.summary() # Print model structure

# Callbacks
output_model_dir = "trained_models"
os.makedirs(output_model_dir, exist_ok=True)

checkpoint_path = os.path.join(output_model_dir, "attention_model_best_epoch.h5")
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_accuracy', # Monitor validation accuracy
                             save_best_only=True,    # Save only the best model
                             mode='max',             # Mode for 'val_accuracy' is 'max'
                             verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', # Monitor validation loss
                               patience=10,         # Epochs to wait for improvement
                               restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored quantity.
                               verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,   # Factor by which learning rate will be reduced: new_lr = lr * factor
                              patience=5,   # Epochs to wait before reducing LR
                              min_lr=1e-6,  # Lower bound on the learning rate
                              verbose=1)

callbacks_list = [checkpoint, early_stopping, reduce_lr]

# Train the Model
print("Starting training...")
history = model.fit(
    train_gen,
    steps_per_epoch=max(1, len(X_train_paths) // BATCH_SIZE), # Ensure at least 1 step
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=max(1, len(X_val_paths) // BATCH_SIZE), # Ensure at least 1 step
    callbacks=callbacks_list,
    verbose=1
)

# Save the final model (which is the best model due to restore_best_weights=True in EarlyStopping)
final_model_path = os.path.join(output_model_dir, "attention_model_final.h5")
model.save(final_model_path)
print(f"Training complete. Final model (best weights) saved as {final_model_path}")
if os.path.exists(checkpoint_path) and final_model_path != checkpoint_path:
    print(f"The explicitly saved best epoch model is at: {checkpoint_path}")


# ────────────────────────────────────────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ────────────────────────────────────────────────────────────────────────────────
acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])

if acc and val_acc and loss and val_loss: # Check if metrics are available
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    history_plot_path = os.path.join(output_model_dir, 'training_history.png')
    plt.savefig(history_plot_path)
    print(f"Training history plot saved to {history_plot_path}")
    # plt.show() # Uncomment to display plot if running in an environment that supports it
else:
    print("Could not plot training history: one or more metrics are missing from history.")

print("--- Training Script Finished ---")