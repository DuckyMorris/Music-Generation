from re import M
from notes_midi import convert_to_midi, read_midi 
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split 
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K
import random

folder  = 'Piano G/'
files= os.listdir(folder)
chords = []
for file in files:
    chords.append(read_midi(folder+file))
chords = np.array(chords, dtype=object)

print("finished line 17")

chord = []
for c in chords:
  for note in c:
    chord.append(note)

freq = dict(Counter(chord))

lim = 25 #chord frequency to be considered a frequent
frequent_notes = []
for a,b in freq.items():
  if b > lim:
    frequent_notes.append(a)

print("finished line 32")


amended = []
for chord in chords:
    temp=[]
    for n in chord:
        if n in frequent_notes:
            temp.append(n)
        amended.append(temp)
amended = np.array(amended, dtype = object)

print("finished line 46")

steps = 32
x = []
y = []

for n in amended:
    for i in range(len(n) - steps):
        input_ = n[i:i + steps]
        output = n[i + steps]
        x.append(input_)
        y.append(output)

x=np.array(x)
y=np.array(y)

print("finished line 63")

unique_x = list(set(x.ravel()))

x_dict = []

for number, n in enumerate(unique_x):
    x_dict.append((n, number))

x_dict = dict(x_dict)

x_progression=[]

for i in x:
    temp=[]
    for j in i:
        temp.append(x_dict[j])

    x_progression.append(temp)

print("finished line 81")

x_progression = np.array(x_progression)

unique_y = list(set(y)) 

y_dict = []

for number, n in enumerate(unique_x):
    y_dict.append((n, number))

y_dict = dict(y_dict)

y_progression = []
for y_ in y:
    y_progression.append(y_dict[y_])
    
y_progression=np.array(y_progression)

trainx, predx, trainy, predy =train_test_split(x_progression,y_progression,test_size=0.2,random_state=0)
print("finished line 100")

K.clear_session()
m = Sequential()
    
m.add(Embedding(len(unique_x), 100, input_length=32,trainable=True)) 

m.add(Conv1D(64,3, padding='causal',activation='relu'))
m.add(Dropout(0.2))
m.add(MaxPool1D(2))
    
m.add(Conv1D(128,3,activation='relu',dilation_rate=2,padding='causal'))
m.add(Dropout(0.2))
m.add(MaxPool1D(2))

m.add(Conv1D(256,3,activation='relu',dilation_rate=4,padding='causal'))
m.add(Dropout(0.2))
m.add(MaxPool1D(2))
          
m.add(GlobalMaxPool1D())
m.add(Dense(256, activation='relu'))

m.add(Dense(len(unique_y), activation='softmax'))   
m.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

m.summary()

print("finished line 126")


mc=ModelCheckpoint('best.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)

print("finished line 131")
e = 25
hist= m.fit(np.array(trainx),np.array(trainy),batch_size=128,epochs=e, validation_data= (np.array(predx),np.array(predy)),verbose=1, callbacks=[mc])

print("finished line 135")

start = random.randint(0,len(predx)-1)

rand = predx[start]

print("finished line 139")
pred=[]

num_notes = 500
for i in range(num_notes):

    rand = rand.reshape(1,steps)

    p  = m.predict(rand)[0]
    y_pred= np.argmax(p,axis=0)
    pred.append(y_pred)

    rand = np.insert(rand[0],len(rand[0]),y_pred)
    rand = rand[1:]

print("finished line 155")
 


x_dict = []

for number, n in enumerate(unique_x):
    x_dict.append((number, n))

x_dict = dict(x_dict)

predicted_notes =[x_dict[i] for i in pred]

convert_to_midi(predicted_notes)

print("Done")