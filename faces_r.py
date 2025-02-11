import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Функция для загрузки обучающей выборки
def load_train(path):
    labels = pd.read_csv(f"{path}/labels.csv")
    datagen = ImageDataGenerator(
        validation_split=0.25,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255
    )
    train_generator = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=f"{path}/final_files",
        x_col="file_name",
        y_col="real_age",
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset='training',
        seed=42
    )
    return train_generator

# Функция для загрузки тестовой выборки
def load_test(path):
    labels = pd.read_csv(f"{path}/labels.csv")
    datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)
    test_generator = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=f"{path}/final_files",
        x_col="file_name",
        y_col="real_age",
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset='validation',
        seed=42
    )
    return test_generator

# Функция для создания модели
def create_model(input_shape):
    backbone = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Функция для обучения модели
def train_model(model, train_data, test_data, batch_size=None, epochs=10, steps_per_epoch=None, validation_steps=None):
    if steps_per_epoch is None: steps_per_epoch = len(train_data)
    if validation_steps is None: validation_steps = len(test_data)
    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        shuffle=True
    )
    return model
