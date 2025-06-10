import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, BatchNormalization, Dropout)
from tensorflow.keras.datasets.mnist import load_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = load_data()
model = None
# 모델이 이미 존재하면 불러오고, 없으면 새로 학습한다.
if tf.io.gfile.exists('mnist_model.h5'):
  model = keras.models.load_model('mnist_model.h5')

def model_train():
  global train_images, train_labels, test_images, test_labels, model
  '''
  mnist 기본 데이터셋은 train_images, test_images가 기본적으로 나뉘어져 있다.
  '''
  print("train_images.shape : ", train_images.shape)
  print("test_images.shape : ", test_images.shape)

  # plt.hist(train_labels, bins=10, rwidth=0.8)
  # plt.title("MNIST Label Distribution")
  # plt.show()

  '''
  train_images 에서 일부분을 validation set으로 사용하기 위해서
  train_images와 train_labels를 분할한다.
  '''
  np.random.seed(42)

  index_list = np.arange(0, len(train_labels))
  validation_index = np.random.choice(index_list, size=5000, replace=False)

  validating_images = train_images[validation_index]
  validating_labels = train_labels[validation_index]

  train_images = np.delete(train_images, validation_index, axis=0)
  train_labels = np.delete(train_labels, validation_index, axis=0)

  plt.hist([train_labels, validating_labels], bins=10, rwidth=0.8, label=['train', 'validating'])
  plt.title("MNIST Label Distribution after splitting")
  plt.show()

  # min-max scaling
  min_key = np.min(train_images)
  max_key = np.max(train_images)
  train_images = (train_images - min_key) / (max_key - min_key)
  validating_images = (validating_images - min_key) / (max_key - min_key)
  test_images = (test_images - min_key) / (max_key - min_key)

  # 모델의 레이어 층을 만듦
  model = keras.models.Sequential([
    keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    Dense(300, activation='relu', name="Hidden_Layer_1"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(200, activation='relu', name="Hidden_Layer_2"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(100, activation='relu', name="Hidden_Layer_3"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(10, activation='softmax', name="Output_Layer")
  ])
  
  optimizer = keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  # 조기 종료 조건 설정
  early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    restore_best_weights=True
  )
  history = model.fit(
    train_images, train_labels, 
    validation_data=(validating_images, validating_labels),
    epochs=100,
    batch_size=5000,
    verbose=1,
    callbacks=[early_stopping]
  )
  
  history_dict = pd.DataFrame(history.history)
  history_dict.plot(figsize=(12, 8), linewidth = 3)
  plt.grid(True)
  
  plt.legend(loc = 'upper right', fontsize = 15)
  plt.title("Learning Curve", fontsize = 20, pad = 20)
  plt.xlabel("Epochs", fontsize = 15)
  plt.ylabel("Variable", fontsize = 15, rotation = 0, loc = 'center', labelpad = 20)
  
  ax = plt.gca()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  
  plt.show()
  
  print(model.evaluate(test_images, test_labels, verbose=1))
  
  model.save('mnist_model.h5')
  
if model is None:
  model_train()
