"""
Código para treinamento da CNN para reconhecimento de sinais da linguagem de sinais americana (ASL)
"""

import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_df = pd.read_csv("..\\sign_language\\dados-libras\\sign_mnist_train\\sign_mnist_train.csv")
test_df = pd.read_csv("..\\sign_language\\dados-libras\\sign_mnist_test\\sign_mnist_test.csv")

train_label=train_df['label']
trainset=train_df.drop(['label'],axis=1)

X_train = trainset.values
X_train = trainset.values.reshape(-1,28,28,1)
print("Quantidade de dados de treino: ", X_train.shape)

test_label=test_df['label']
X_test=test_df.drop(['label'],axis=1)
X_test=X_test.values.reshape(-1,28,28,1)
print("Quantidade de dados de teste: ", X_test.shape)


# Transformação dos dados de treinamento e teste para o formato de matriz binária
lb=LabelBinarizer()
y_train=lb.fit_transform(train_label)
y_test=lb.fit_transform(test_label)



# Aumento de dados de treinamento com ImageDataGenerator para evitar overfitting e gerar mais dados de treinamento
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 0,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

X_test=X_test/255

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

# Numero de classes (simbolos)
num_classes = 24

# Modelo da CNN com 3 camadas convolucionais e MaxPooling com Dropout e BatchNormalization
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.BatchNormalization(input_shape=(28,28,1))) # Normalização dos dados de entrada 
# Primeira camada convolucional
model.add(tf.keras.layers.Conv2D(filters= 128 ,kernel_size = (5, 5), activation='relu', strides=1, padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same'))
model.add(tf.keras.layers.Dropout(0.15))
# Segunda camada convolucional
model.add(tf.keras.layers.Conv2D(filters= 128,kernel_size = (3, 3), strides=1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.15))
# Terceira camada convolucional
model.add(tf.keras.layers.Conv2D(filters= 64,kernel_size = (2, 2), strides=1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=1))
model.add(tf.keras.layers.Dropout(0.15))
#A CAMADA TOTALMENTE CONECTADA
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
model.summary()


# Compilação do modelo
# Otimizador: Adam (Adaptive Moment Estimation) com taxa de aprendizado padrão de 0.001
# Função de perda: Cross Entropy (entropia cruzada) para classificação multiclasse 
# Métrica de avaliação: Acurácia
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Treinamento do modelo 
history = model.fit(train_datagen.flow(X_train,y_train,batch_size=200),
        epochs = 100,
        validation_data=(X_test,y_test),
        callbacks=[early_stop],
        shuffle=1
         )


# Gráficos da evolução do treinamento e validação
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Treinamento e Validação em Relação ao Tempo')
plt.xlabel('Épocas')
plt.ylabel('Perdas')
plt.legend(['Treino', 'Validação'])
fig = plt.gcf()
fig.savefig('perda.jpg', dpi=100)
plt.show()

model.save('model_libras.h5')
historico = pd.DataFrame(history.history)
historico.to_csv('Historico.csv')

(ls,acc)=model.evaluate(x=X_test,y=y_test)
print('MODEL ACCURACY = {}%'.format(acc*100))

# matriz de confusão
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_df['label']
cm = confusion_matrix(y_true, y_pred)
# plot da matriz de confusão
fig = plt.figure(figsize=(10, 10))

display = ConfusionMatrixDisplay(cm)
# ploting the Matrix
display.plot()
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
#plt.xticks(ticks = np.arange(num_classes), labels = ["A", "B", "C", "D",  ] )

plt.savefig('matriz_confusao.jpg', dpi=100)

print(cm)



