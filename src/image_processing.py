import numpy as np 
import pandas as pd
from pathlib import Path 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

letters = 'ABCD'
num_classes = len(letters)


def generate_df(image_dir, label, exention=r'*.png'):
  
    filepaths = pd.Series(list(image_dir.glob(exention)), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)
    return df

def combine_dataframes(path_folder, letters=letters):
    dataframes = {}
    extensao_lista = ['*.png', '*.jpeg', '*.jpg']
    
    for letra in letters:
        dataframes[letra] = pd.DataFrame()
        for extension in extensao_lista:
            diretorio_letra = Path(path_folder) / letra
            df = generate_df(image_dir=diretorio_letra, label=letra, exention=extension)
            dataframes[letra] = pd.concat([dataframes[letra], df], axis=0, ignore_index=True)

    list_of_dfs = list(dataframes.values())
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    return combined_df

path_folder_train = "/home/cilab_user/sign_language/dados-libras/train" 
path_folder_test = "/home/cilab_user/sign_language/dados-libras/test" 

train_df= combine_dataframes(path_folder_train, letters=letters)
test_df = combine_dataframes(path_folder_test, letters=letters) 

print(train_df['Label'].value_counts())
print(test_df['Label'].value_counts())


train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./225,
    shear_range=0.2,
    rotation_range = 0,
    zoom_range=0.5,
    fill_mode='nearest'
)


test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./225
)

target_size = (128,128)
class_mode = 'categorical'
shuffle = True
batch_size = 64

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col = 'Filepath',
    y_col =  'Label',
    target_size = target_size,
    color_mode = 'grayscale',
    class_mode = class_mode,
    batch_size = batch_size,
    shuffle = shuffle,
    subset = 'training'
)

val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col = 'Filepath',
    y_col = 'Label',
    target_size = target_size,
    color_mode = 'grayscale',
    class_mode = class_mode,
    batch_size = batch_size,
    shuffle = shuffle,
    subset = 'validation'
)

test_data = test_gen.flow_from_dataframe(
    test_df,
    x_col = 'Filepath',
    y_col = 'Label',
    target_size = target_size,
    color_mode = 'grayscale',
    class_mode = class_mode,
    batch_size = batch_size,
    shuffle = shuffle
    
)


shape = train_data.image_shape

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(128,kernel_size=(5,5), strides=1,padding='same',activation='relu',input_shape=shape))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(tf.keras.layers.Conv2D(64,kernel_size=(2,2), strides=1,activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPool2D((2,2),2,padding='same'))
model.add(tf.keras.layers.Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPool2D((2,2),2,padding='same'))        
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512,activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.Dense(units=num_classes,activation='softmax'))
model.summary()

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(
    train_data,
    validation_data = test_data,
    epochs = 100,
    shuffle=True,
    batch_size = batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Treinamento e Validação em Relação ao Tempo')
plt.xlabel('Épocas')
plt.ylabel('Perdas')
plt.legend(['Treino', 'Validação'])
plt.show()

# Testando o modelo
print(model.evaluate(test_data))
# confusion matrix
y_pred = model.predict(test_data)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_data.classes
cm = confusion_matrix(y_true, y_pred)
print(cm)

