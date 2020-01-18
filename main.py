import os
import nltk
import json
import random
import pickle
import tflearn
import numpy as np
import pyttsx3 as tts
import tensorflow as tf
import webbrowser as wb
import speech_recognition as sr
from nltk.stem.lancaster import LancasterStemmer


class Chatbot:
    def __init__(self):
        self.machine = tts.init()
        voices = self.machine.getProperty('voices')
        self.machine.setProperty('voice', voices[1].id)

        self.recognizer = sr.Recognizer()

        self.stemmer = LancasterStemmer()

        chrome_path = 'C:\\Program Files (x86)\\Google\Chrome\\Application\\chrome.exe'
        wb.register('chrome', None, wb.BackgroundBrowser(chrome_path))

        with open('data/intents.json') as f:
            self.data = json.load(f)

        dirs = os.listdir('data')
        dir_len = len(dirs)

        self.words = []
        self.classes = []
        self.training = []
        self.output = []

        if dir_len == 1:
            patterns = []
            tags = []

            for intent in self.data['intents']:
                for pattern in intent['patterns']:
                    tokenized_words = nltk.word_tokenize(pattern)
                    self.words.extend(tokenized_words)
                    patterns.append(tokenized_words)
                    tags.append(intent['tag'])

                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

            self.words = [self.stemmer.stem(i.lower()) for i in self.words if i != '?']
            self.words = sorted(list(set(self.words)))

            self.classes = sorted(self.classes)

            output_default = [0 for i in range(len(self.classes))]

            for x, doc in enumerate(patterns):
                bag_of_words = []
                words_from_pattern = [self.stemmer.stem(i) for i in doc]

                for w in self.words:
                    if w in words_from_pattern:
                        bag_of_words.append(1)
                    else:
                        bag_of_words.append(0)

                output_row = output_default[:]
                output_row[self.classes.index(tags[x])] = 1

                self.training.append(bag_of_words)
                self.output.append(output_row)

            self.training = np.array(self.training)
            self.output = np.array(self.output)

            tf.reset_default_graph()

            self.net = tflearn.input_data(shape=[None, len(self.training[0])])
            self.net = tflearn.fully_connected(self.net, 8)
            self.net = tflearn.fully_connected(self.net, 8)
            self.net = tflearn.fully_connected(self.net, len(self.output[0]), activation='softmax')

            self.net = tflearn.regression(self.net)

            self.model = tflearn.DNN(self.net)
            self.model.fit(self.training, self.output, n_epoch=2000, batch_size=8, show_metric=True)
            self.model.save('data/model.tflearn')

            with open('data/data.pickle', 'wb') as f:
                pickle.dump((self.words, self.classes, self.training, self.output), f)

        else:
            with open('data/data.pickle', 'rb') as f:
                self.words, self.classes, self.training, self.output = pickle.load(f)

            tf.reset_default_graph()

            self.net = tflearn.input_data(shape=[None, len(self.training[0])])
            self.net = tflearn.fully_connected(self.net, 8)
            self.net = tflearn.fully_connected(self.net, 8)
            self.net = tflearn.fully_connected(self.net, len(self.output[0]), activation='softmax')

            self.net = tflearn.regression(self.net)

            self.model = tflearn.DNN(self.net)
            self.model.load('data/model.tflearn')

    def get_audio(self):
        with sr.Microphone() as src:
            audio = self.recognizer.listen(src)

        try:
            return self.recognizer.recognize_google(audio)
        except:
            return 'Seems like you are offline! Connect to the internet and try again!'

    def open_google(self, query):
        wb.get('chrome').open_new_tab('https://www.google.com/search?q=' + query.replace(' ', '+'))

    def generate_word_bag(self, sentence):
        bag = [0 for i in range(len(self.words))]

        words_in_sentence = nltk.word_tokenize(sentence)
        words_in_sentence = [self.stemmer.stem(word.lower()) for word in words_in_sentence]

        for word in words_in_sentence:
            for i, w in enumerate(self.words):
                if w == word:
                    bag[i] = 1

        return np.array(bag)

    def chat(self):
        while True:
            user = input('user : ')

            results = self.model.predict([self.generate_word_bag(user)])
            highest_probability = np.argmax(results)

            confidence = results[0][highest_probability]

            if confidence > 0.9:
                tag = self.classes[highest_probability]

                if tag == 'math':
                    user = user.lower()
                    user = user.replace('times', '*')
                    user = user.replace('plus', '+')
                    user = user.replace('minus', '-')
                    user = user.replace('by', '/')
                    user = user.replace('mod', '%')
                    user = user.replace('percent of', '/100*')

                    for i in user:
                        if i not in (
                                '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-', 'x', '/', '%', '*', '(',
                                ')'):
                            user = user.replace(i, '')

                        elif i == 'x':
                            user = user.replace(i, '*')

                    ans = f'It is {eval(user)}!'

                    print(ans)
                    self.machine.say(ans)

                else:
                    responses = []
                    for cls in self.data['intents']:
                        if cls['tag'] == tag:
                            responses = cls['responses']

                    ans = random.choice(responses)
                    print(ans)
                    self.machine.say(ans)

                    if tag == 'goodbye' :
                        self.machine.runAndWait()
                        break

            else:
                print('Sorry! Didn\'t get ya!...Looking in the internet!')
                self.machine.say('Sorry! Didn\'t get ya!...Looking in the internet!')
                self.open_google(user)

            self.machine.runAndWait()


may = Chatbot()
may.chat()
