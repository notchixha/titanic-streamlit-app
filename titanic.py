import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import pickle


st.title('PREDICT YOUR SURVIVAL ON THE TITANIC')
img = Image.open('titanic.jpg')
st.image(img,width=600, channels='RGB',caption=None)



model = load_model('titanics.h5')
history = pickle.load(open('training_history','rb'))

def survival():
    Pclass = st.sidebar.slider('Pclass',1,3)
    sex = st.sidebar.selectbox('Sex', ('Male','Female'))
    age = st.sidebar.slider('Age',0,80)
    n_siblings_spouses = st.sidebar.slider('N_siblings/spouses',0,3)
    n_parents_children = st.sidebar.slider('N_parents/children',0,3)
    fair = st.sidebar.slider('Fair',0,200)
    
    if sex == 'Male':
        sex = 1
    else:
        sex = 0

    data = [[Pclass, sex, age, n_siblings_spouses,n_parents_children,fair]]
    data = tf.constant(data)
    return data

survive_or_not=survival()
prediction = model.predict(survive_or_not, steps=1)
pred = [round(x[0]) for x in prediction]

if pred ==[0]:
    st.write('You did not survive')
else:
    st.write('You survived')


def plot_data():
    loss_train=history['loss']
    loss_val = history['val_loss']
    epochs = range(1,120)
    fig, ax = plt.subplots()
    ax.scatter([0.25],[0.25])
    plt.plot(epochs, loss_train,'g', label = 'Training Loss')
    plt.plot(epochs, loss_val,'b', label = 'Validation Loss')
    plt.title('Trainig and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    st.pyplot(fig)
    
    loss_train =history['acc']
    loss_val = history['val_acc']
    epochs = range(1,120)
    fig, ax = plt.subplots()
    ax.scatter([1], [1])
    plt.plot(epochs, loss_train, 'g', label='Training Accuracy')
    plt.plot(epochs, loss_val, 'b', label='Validation Accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    st.pyplot(fig)
    
plot_data()