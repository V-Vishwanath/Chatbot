import os
import nltk
import json
import random
import pickle
import tflearn
import platform
import numpy as np
import pyttsx3 as tts
import webbrowser as wb
import pyautogui as gui
import tensorflow as tf
from subprocess import Popen
import speech_recognition as sr
from nltk.stem.lancaster import LancasterStemmer


dynamic_actions = [
    'make-note',
    'get-note',
    'math'
]

class Chatbot:
    def __init__(self):

        self.currdir = os.getcwd()
        self.notes_dir = os.path.join(self.currdir, 'Notes')

        self.OS = platform.system()
        self.chrome_path = ''

        self.machine = tts.init()

        if self.OS == 'Windows' :
            voices = self.machine.getProperty('voices')
            self.machine.setProperty('voice', voices[1].id)
            chrome_path = 'C:\\Program Files (x86)\\Google\Chrome\\Application\\chrome.exe'
        
        else :
            self.machine.setProperty('voice', 'english+f5')
            self.machine.setProperty('rate', 180)
            chrome_path = '/usr/bin/google-chrome-stable'

        wb.register('chrome', None, wb.BackgroundBrowser(chrome_path))

        self.recognizer = sr.Recognizer()

        self.stemmer = LancasterStemmer()
        
        self.machine.say('Initializing!')
        self.machine.runAndWait()

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
            self.model.fit(self.training, self.output, n_epoch=1000, batch_size=8, show_metric=True)
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
            return ''

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

    def predict(self, user) :
        results = self.model.predict([self.generate_word_bag(user)])
        highest_probability = np.argmax(results)

        return (results[0][highest_probability], self.classes[highest_probability])


    def get_responses(self, tag) :
        responses = []
        for cls in self.data['intents']:
            if cls['tag'] == tag:
                responses = cls['responses']
                break 

        return responses

    def speak(self, text) :
        print(f'Bot : {text}')
        self.machine.say(text)
        self.machine.runAndWait()

    # Action to perform
    def make_note(self) :
        self.speak('What do you like to call this note?')
        name = self.get_audio() + '.txt'

        name = os.path.join(self.notes_dir, name)

        self.speak('Okay...What do you want me to note down?')

        note = ''
        while True :
            note = self.get_audio()
            if note != '' :
                break

        print(f'You : {note}')

        with open(name, 'a') as f :
            f.write(note + '\n')

        while True :
            self.speak('Is that it?')
            done = self.get_audio().lower()
            print(f'You : {done}')

            if 'yeah' in done or 'yes' in done or 'that\'s it' in done :
                break

            self.speak('Okay! Go ahead!')

            note = ''
            while True :
                note = self.get_audio()
                if note != '' :
                    break
                
            print(f'You : {note}')

            with open(name, 'a') as f :
                f.write(note + '\n')

        self.speak('Done!')

        if self.OS == 'Windows' :
            Popen(['notepad.exe', name])
        else :
            Popen(['gedit', name], stdin=open(os.devnull, 'r'))


    def get_note(self) :
        self.speak('Which note would you like to open? Just say the number..')
        notes = os.listdir(self.notes_dir)

        for i in range(len(notes)) :
            print(f'{i+1}) {notes[i]}')

        note = self.get_audio()

        try :
            Popen(['gedit', os.path.join(self.notes_dir, notes[int(note)])], stdin=open(os.devnull, 'r'))
            self.speak('Okay!')
        except :
            self.speak('Sorry! Couldn\'t get that!')


    def perform_action(self, tag) :
        if tag == 'make-note' :
            self.make_note()

        elif tag == 'get-note' :
            self.get_note()

    def chat(self):
        self.machine.say('Go ahead, I\'m listening!')
        self.machine.runAndWait()
        
        while True:
            user = self.get_audio()

            if user == '' :
                continue

            print(f'you : {user}')
            confidence, tag = self.predict(user)

            if confidence > 0.9:
                if tag in dynamic_actions :
                    self.perform_action(tag)
                    
                else:
                    ans = random.choice(self.get_responses(tag))
                    self.speak(ans)

                    if tag == 'goodbye' :
                        break

            else:
                self.speak('Sorry! Didn\'t get you! Looking in the internet!')
                self.open_google(user)


may = Chatbot()
may.chat()
