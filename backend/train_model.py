import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from datetime import datetime

# Configuration
DATASET_PATH = 'data/1'  # Base directory containing train and test folders
MODEL_SAVE_PATH = 'models/emotion_model.keras'
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# Emotion labels (must match folder names)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def verify_dataset_structure(dataset_path):
    """
    Verify the dataset structure and print statistics
    """
    print("Verifying dataset structure...")
    
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    
    if not os.path.exists(train_path):
        print(f"Error: Training directory not found at {train_path}")
        return False
    
    if not os.path.exists(test_path):
        print(f"Error: Test directory not found at {test_path}")
        return False
    
    print(f"\n{'='*70}")
    print("Dataset Structure:")
    print(f"{'='*70}")
    
    # Count images in each category
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_path, split)
        print(f"\n{split.upper()} SET:")
        total = 0
        
        for emotion in EMOTION_LABELS:
            emotion_path = os.path.join(split_path, emotion)
            if os.path.exists(emotion_path):
                count = len([f for f in os.listdir(emotion_path) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {emotion.capitalize():12s}: {count:5d} images")
                total += count
            else:
                print(f"  {emotion.capitalize():12s}: Missing!")
                return False
        
        print(f"  {'Total':12s}: {total:5d} images")
    
    print(f"\n{'='*70}\n")
    return True

def create_data_generators(dataset_path):
    """
    Create data generators for training and validation
    """
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    # Validation data (only rescaling, no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True,
        classes=EMOTION_LABELS  # Ensures consistent label ordering
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False,
        classes=EMOTION_LABELS
    )
    
    return train_generator, test_generator

def create_cnn_model():
    """
    Create CNN model architecture for emotion recognition
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output Layer
        Dense(7, activation='softmax')
    ])
    
    return model

def plot_training_history(history, save_path='plots/training_history.png'):
    """
    Plot training history (accuracy and loss)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")
    plt.close()

def plot_confusion_matrix(model, test_generator, save_path='plots/confusion_matrix.png'):
    """
    Plot confusion matrix
    """
    # Reset generator
    test_generator.reset()
    
    # Get predictions
    steps = len(test_generator)
    y_pred = model.predict(test_generator, steps=steps, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label.capitalize() for label in EMOTION_LABELS],
                yticklabels=[label.capitalize() for label in EMOTION_LABELS],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()
    
    # Print classification report
    print("\n" + "="*70)
    print("Classification Report:")
    print("="*70)
    print(classification_report(y_true, y_pred_classes, 
                                target_names=[label.capitalize() for label in EMOTION_LABELS]))

def train_model():
    """
    Main training function
    """
    print("\n" + "="*70)
    print("  FACIAL EMOTION RECOGNITION - CNN MODEL TRAINING")
    print("="*70)
    print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Verify dataset
    if not verify_dataset_structure(DATASET_PATH):
        print("\n❌ Dataset verification failed!")
        return None, None
    
    # Create data generators
    print("Creating data generators...")
    train_generator, test_generator = create_data_generators(DATASET_PATH)
    
    print(f"\n✓ Found {train_generator.samples} training images")
    print(f"✓ Found {test_generator.samples} validation images")
    print(f"✓ Number of classes: {train_generator.num_classes}")
    
    # Create model
    print("\n" + "="*70)
    print("Creating CNN model...")
    print("="*70)
    model = create_cnn_model()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Calculate steps
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = test_generator.samples // BATCH_SIZE
    
    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Steps per Epoch: {steps_per_epoch}")
    print(f"  Validation Steps: {validation_steps}")
    print(f"  Max Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "="*70)
    print("Evaluating model on test set...")
    print("="*70)
    
    test_loss, test_accuracy = model.evaluate(
        test_generator,
        steps=validation_steps,
        verbose=0
    )
    
    print(f"\n📊 Final Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Plot training history
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70 + "\n")
    
    plot_training_history(history)
    plot_confusion_matrix(model, test_generator)
    
    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")
    
    print("\n" + "="*70)
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return model, history

if __name__ == '__main__':
    try:
        # Train the model
        model, history = train_model()
        
        if model is not None:
            print("\n" + "="*70)
            print("✓ TRAINING COMPLETE!")
            print("="*70)
            print("\nNext steps:")
            print("  1. Check 'models/emotion_model.h5' - your trained model")
            print("  2. Check 'plots/' folder for training visualizations")
            print("  3. Run 'python app.py' to start the Flask backend")
            print("  4. Connect your React frontend to http://localhost:5000")
            print("\n" + "="*70 + "\n")
        else:
            print("\n❌ Training failed. Please check the errors above.")
            
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()