#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 02:03:08 2018

@author: chandan
"""
"""
Contains whole code for chatbot. 

1. How to build it  
2. how to train it 
3. How we test it 

"""
# Building a ChatBot with DeepNLP


#Importing the libraries 
# numpy - to work with arrays. tensorflow - Deeplearning re - to clean the text. time - to measure traiing time


import numpy as np
import tensorflow as tf
import re 
import time


############################# PART 1 - DATA PREPROCESSING  ####################### 

#Importing the dataset

lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

#creating a dictionary that maps each line and its id

id2line = {}

for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
        
#Creating a list of all of the cobersation
        
conversations_ids = []

for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ","")
    conversations_ids.append(_conversation.split(','))


#Getting separately the questions and the answers
    
questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
        
#Cleaning the texts
        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am", text)
    text = re.sub(r"he's","he is", text)
    text = re.sub(r"she's","she is", text)    
    text = re.sub(r"that's","that is", text)
    text = re.sub(r"what's","what is", text)
    text = re.sub(r"where's","where is", text)
    text = re.sub(r"\'ll"," will", text)
    text = re.sub(r"\'ve"," have", text)
    text = re.sub(r"\'re"," are", text)
    text = re.sub(r"\'d"," would", text)
    text = re.sub(r"won't","will not", text)
    text = re.sub(r"can't","cannot", text)
    text = re.sub(r"\'bout","about", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]","",text)
    return text


#cleaning the questions
    
clean_questions = []

for question in questions:
    clean_questions.append(clean_text(question))
    
    


#clean the answers
    
clean_answers = []

for answer in answers:
    clean_answers.append(clean_text(answer))
    
    
#creating a dictionary that maps each word to its number of occurences
wordCount = {}

for question in clean_questions:
    for word in question.split():
        if word not in wordCount:
            wordCount[word] = 1
        else:
            wordCount[word] += 1
        
for answer in clean_answers: 
    for word in answer.split():
        if word not in wordCount:
            wordCount[word] = 1
        else:
            wordCount[word] += 1

#Creating two dictionary that map the questions words and the answers words to a unique integer 
            
threshold = 20
questionswords2int = {}
word_number = 0
for word, count in wordCount.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1


answerswords2int = {}
word_number = 0
for word, count in wordCount.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1 
        
#Adding the last tokens to these two dictionaries 
        
tokens = ['<PAD>', '<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1
        
#Creating the inverse dictionary of the answersword2int dictionary 
answersint2word = {w_i: w for w, w_i in answerswords2int.items()}

#Adding EOS string to every answers

for i in range (len (clean_answers)):
    clean_answers[i] += ' <EOS>'
    
    
#Translating all the questions and the answers into integers
#and replacing all the words that were filtered by <out>

questionstoint = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word ])
    questionstoint.append(ints)
    

answerstoint = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word ])
    answerstoint.append(ints)
    
#Sorting the questions and answers by the length of questions
    
sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1,25 + 1):
    for i in enumerate(questionstoint):
        if len(i[1]) == length:
            sorted_clean_questions.append(questionstoint[i[0]])
            sorted_clean_answers.append(answerstoint[i[0]])