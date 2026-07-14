import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_DIR = r'C:\Users\jhoon\Downloads\proyecto_final_IA\archive (1)\train'
TEST_DIR  = r'C:\Users\jhoon\Downloads\proyecto_final_IA\archive (1)\test'

# ==============================
# CLASES SELECCIONADAS
# ==============================
# Carpetas disponibles:
# angry, disgusting, fearful, happy, neutral, sad, surprised
#
# En esta prueba quitamos fearful y usamos surprised
SELECTED_CLASSES = ['angry', 'happy', 'neutral', 'sad', 'surprised']

# ==============================
# VERIFICAR CARPETAS
# ==============================
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"No existe la carpeta de entrenamiento: {TRAIN_DIR}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"No existe la carpeta de prueba: {TEST_DIR}")

for clase in SELECTED_CLASSES:
    train_class_path = os.path.join(TRAIN_DIR, clase)
    test_class_path = os.path.join(TEST_DIR, clase)

    if not os.path.exists(train_class_path):
        raise FileNotFoundError(f"No existe la clase en train: {train_class_path}")

    if not os.path.exists(test_class_path):
        raise FileNotFoundError(f"No existe la clase en test: {test_class_path}")

# ==============================
# DATA AUGMENTATION
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    zoom_range=0.05,
    width_shift_range=0.03,
    height_shift_range=0.03,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    classes=SELECTED_CLASSES,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    classes=SELECTED_CLASSES,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Clases:", train_data.class_indices)

# ==============================
# CLASS WEIGHTS
# ==============================
class_counts = train_data.classes
class_weights = {}

unique, counts = np.unique(class_counts, return_counts=True)
total = sum(counts)

for i, count in zip(unique, counts):
    class_weights[i] = total / (len(unique) * count)

print("Class weights:", class_weights)

# ==============================
# MODELO CNN PARA 5 CLASES
# ==============================
model = models.Sequential([

    layers.Input(shape=(48, 48, 1)),

    # Bloque 1
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),

    # Bloque 2
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Bloque 3
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),

    # Bloque 4 ligero
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.35),

    # 5 clases: angry, happy, neutral, sad, surprised
    layers.Dense(5, activation='softmax')
])

# ==============================
# COMPILACIÓN
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

model.summary()

# ==============================
# CALLBACKS
# ==============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_emotion_model_5classes_surprised.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# ==============================
# ENTRENAMIENTO
# ==============================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ==============================
# GUARDAR MODELO FINAL
# ==============================
model.save("emotion_model_5classes_surprised.keras")

print("Entrenamiento terminado.")
print("Modelo guardado como emotion_model_5classes_surprised.keras")

# ==============================
# MATRIZ DE CONFUSIÓN
# ==============================
test_data.reset()

predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

class_names = list(test_data.class_indices.keys())

print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Clase real")
plt.colorbar()

plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()