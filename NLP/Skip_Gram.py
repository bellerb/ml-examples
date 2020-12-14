#Import libraries
import re
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Conv1D, MaxPooling1D
from tensorflow.python.keras.callbacks import TensorBoard

#Import custom classes
[sys.path.insert(0,f) for f in ['core','ml']] #Add custom class folder locations
from gym import train
from brain import networks,learn
from models import models,brain
from encoder import encode_data

#Import data
date = datetime.now()
settings = load.readData('static/settings.json') #Read user settings
locations = settings['locations'] #Location of where data located
txt_file = open('0to1.txt','r',encoding="utf8") #Read txt for training data
content = txt_file.read() #Import text data

#Process data
sentences = [s.strip() for s in re.split('[.?!]',content) if len(s) > 0]
word_bank = []
for s in sentences:
    for w in s.split(' '):
        if w not in word_bank:
            word_bank.append(w)

#Build training data
window = 2
data = {'context':[],'target':[]}
for s in sentences:
    words = s.split(' ')
    for i,w in enumerate(words):
        context1 = [encode_data.one_hot(c,word_bank) for c in words[i-window:i]]
        while len(context1) < window:
            context1.insert(0,[0]*len(word_bank))
        context2 = [encode_data.one_hot(c,word_bank) for c in words[i+1:i+window+1]]
        while len(context2) < window:
            context2.append([0]*len(word_bank))
        data['target'].append(encode_data.one_hot(w,word_bank))
        data['context'].append(context1+context2)

data = {'target':np.array(data['target'][10:]),'context':np.array(data['context'][10:])}
test = {'target':np.array(data['target'][:10]),'context':np.array(data['context'][:10])}

#Build model
name = 'SKIP-GRAM_example'
layers = [{'Size':100,'Activation':'relu'},
          {'Size':75,'Activation':'relu'},
          {'Size':len(word_bank),'Activation':'softmax'}]
#Build model --------------------------------->
model = Sequential()
model.add(Flatten()) #Must flatten layer if going from 2D to 1D (dense)
model.add(Dense(layers[0]['Size'],input_shape=data['context'].shape[1:]))
model.add(Activation(layers[0]['Activation']))
for l in layers[1:]:
    model.add(Dropout(0.5))
    model.add(Dense(l['Size']))
    model.add(Activation(l['Activation']))
#Train model --------------------------------->
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,y,epochs=100,validation_split=0.3,callbacks=[tensorboard])

#Save & log new model ------------------------>
if os.path.exists('{}/logs/models.csv'.format(locations['main'])):
    data = load.readData('{}/logs/models.csv'.format(locations['main']))
else:
    data = pd.DataFrame()
info = name.split('-')
type = info[0]
purpose = info[1]
id = info[2]
end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
result = {
            'ID':id,
            'Name':name,
            'Purpose':purpose,
            'Type':type,
            'Dx':'{}-X.pickle'.format(name),
            'Dy':'{}-Y.pickle'.format(name),
            'Std':'',
            'Mean':'',
            'Start':date.strftime('%Y-%m-%d %H:%M:%S'),
            'End':end,
            'Status':True
         }
data = data.append(result,ignore_index=True)
data.to_csv('{}/logs/models.csv'.format(location),index=False)

model.save('{}/ml/models/{}.model'.format(location,name).replace('/','\\'))


'''
#load model
word2vec = tf.keras.models.load_model('{}/MRP/ml/models/{}.model'.format(location,name).replace('/','\\'))
test_word = word_bank[5]
word_vector = encode_data.word_vector(test_word,word_bank,word2vec)
print('{} = {}'.format(test_word,word_vector))
'''

'''
#Test model
for index in range(10):
    test_input = test['context'][index]
    input = test_input.reshape(1,test_input.shape[0],test_input.shape[1])
    pred = word2vec.predict(input)

    #Print result
    clean = [int(round(i,0)) for i in pred[0]]
    user_input = ''
    for x,w in enumerate(test_input):
        if x == 2:
            if len(input) > 0:
                user_input += ' __'
            else:
                user_input += '__'
        text = ' '.join([word_bank[i] for i,state in enumerate(w) if state == 1])
        if len(text) > 0 and len(user_input) > 0:
            user_input += ' ' + text
        else:
            user_input += text
    print('input = {}'.format(user_input))
    print('result = {}\n'.format([word_bank[i] for i,state in enumerate(clean) if state == 1][0]))
'''
