import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import random
import pickle

with open("data.pickle","rb") as f:
      #save all of these data in our pickle file
     data, words, labels, training, output = pickle.load(f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.load("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1         
    return np.array(bag)

def chat(msg):
        print(msg)
        inp = msg
        if inp.lower() == "quit":
            #break
           print("really working")
        
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results) 
        tag = labels[results_index]
       
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']                  
            return (random.choice(responses))
        else:
            answer="Sorry, I didn't get that. Try again or ask a different question."
            return (answer)