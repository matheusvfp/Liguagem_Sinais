import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

num_classes=24
test_df = pd.read_csv("..\\sign_language\\dados-libras\\sign_mnist_test\\sign_mnist_test.csv")

test_label=test_df['label']
X_test=test_df.drop(['label'],axis=1)
X_test=X_test.values.reshape(-1,28,28,1)


# Transformação dos dados de treinamento e teste para o formato de matriz binária
lb=LabelBinarizer()
y_test=lb.fit_transform(test_label)

X_test=X_test/255

historico = pd.read_csv('Historico.csv')

# Gráficos da evolução do treinamento e validação
plt.plot(historico['loss'])
plt.plot(historico['val_loss'])
plt.title('Treinamento e Validação em Relação ao Tempo')
plt.xlabel('Épocas')
plt.ylabel('Perdas')
plt.legend(['Treino', 'Validação'])
fig = plt.gcf()
fig.savefig('perda.jpg', dpi=100)
plt.show()
model = tf.keras.models.load_model('model_libras.h5')

(ls,acc)=model.evaluate(x=X_test,y=y_test)
print('MODEL ACCURACY = {}%'.format(acc*100))
plt.rcParams.update({'font.size': 7})

# matriz de confusão
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_df['label']
cm = confusion_matrix(y_true, y_pred)
new_cm= np.delete(cm, 9, axis=0)
new_cm=np.delete(new_cm, 24, axis=1)
# plot da matriz de confusão
fig = plt.figure(figsize=(20, 20))

display = ConfusionMatrixDisplay(new_cm)
# ploting the Matrix
display.plot()
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
plt.xticks(ticks=np.arange(num_classes), labels=class_labels, fontsize=8)
plt.yticks(ticks=np.arange(num_classes), labels=class_labels, fontsize=8)
plt.show()
#fig = plt.gcf()
#fig.savefig('matriz_confusao.jpg', dpi=100)

print(cm)


