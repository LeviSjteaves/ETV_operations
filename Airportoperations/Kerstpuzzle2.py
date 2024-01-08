# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:43:11 2023

@author: levi1
"""
import random
import math
import requests
import string

def download_dutch_dictionary():
    url = "https://raw.githubusercontent.com/oprogramador/most-common-words-by-language/master/src/resources/dutch.txt"
    response = requests.get(url)

    if response.status_code == 200:
        words = response.text.splitlines()
        return words
    else:
        print("Failed to download the Dutch dictionary.")
        return []

# Download the Dutch dictionary
dutch_dictionary = download_dutch_dictionary()
word_dict_or = set(dutch_dictionary)
word_dict = set(filter(lambda word: len(word) <= 13, word_dict_or))
word_dict.add('moeilijkheid')

# Function to calculate the score
def calculate_score(words):
    score = 0
    for word in words.values():
        if word in word_dict:
            score += 1
    return score

# Simulated Annealing algorithm with constrained letters
def simulated_annealing(initial_state, puzzle, temperature, cooling_rate, iterations, constrained_letters):
    current_state = initial_state.copy()
    current_words = {}

    for k in puzzle:
        wordlist = [current_state[puzzle[k][i]] for i in range(len(puzzle[k]))]
        current_words[k] = ''.join(wordlist)

    current_score = calculate_score(current_words)

    max_score = current_score
    solutions_with_max_score = [(current_state, current_words)]

    for _ in range(iterations):
        # Generate a neighbor state by swapping two random letters (except constrained letters)
        neighbor_state = current_state.copy()
        idx1, idx2 = random.sample([i for i in range(len(neighbor_state)) if i not in constrained_letters], 2)
        neighbor_state[idx1], neighbor_state[idx2] = neighbor_state[idx2], neighbor_state[idx1]

        # Recalculate the words and score for the neighbor state
        neighbor_words = {}
        for k in puzzle:
            wordlist = [neighbor_state[puzzle[k][i]] for i in range(len(puzzle[k]))]
            neighbor_words[k] = ''.join(wordlist)

        neighbor_score = calculate_score(neighbor_words)

        # Accept the neighbor if it improves the score or with a probability
        if neighbor_score > current_score or random.random() < math.exp((neighbor_score - current_score) / temperature):
            current_state = neighbor_state
            current_words = neighbor_words
            current_score = neighbor_score

            # Check if the current solution has a higher score
            if current_score > max_score:
                max_score = current_score
                solutions_with_max_score = [(current_state, current_words)]
            elif current_score == max_score:
                solutions_with_max_score.append((current_state, current_words))

        # Cool the temperature
        temperature *= 1 - cooling_rate

    return solutions_with_max_score, max_score

l = ['0'] * 27

# Initial random state
l[1]='p'
l[2]='x'
l[3]='c'
l[4]='v'
l[5]='q'
l[6]='u'
l[7]='j'#
l[8]='n'#
l[9]='z'#
l[10]='a'
l[11]='h'#
l[12]='i'#
l[13]='k'#
l[14]='o'#
l[15]='l'#
l[16]='e'#
l[17]='m'#
l[18]='g'#
l[19]='f'#
l[20]='d'#
l[21]='t'#
l[22]='r'#
l[23]='y'
l[24]='w'#
l[25]='b'#
l[26]='s'#

initial_state = list(''.join(l))
constrained_letters = [4,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26]


# Define the puzzle structure (graph of connected boxes)
puzzle = {
    'Y2': [8, 10],                                 #Yellow 
    'O1': [4, 22, 10, 10, 18],                         #orange Sure
    'O2': [24, 16, 15, 13, 16],                        #orange Sure
    
    #'P1': [7, 10, 10, 22, 26, 21],                    #pink 
    #'P2': [20, 16, 22],                               #pink 
    #'P3': [13, 16],                                   #pink 
    'P4': [7, 10, 10, 22],                             #pink 
    'P5': [13, 16, 22, 26, 21],                        #pink 
    'P6': [20,16],
    
    'G1': [17, 16, 16],                                #green Sure 
    'G2': [26, 21, 10, 5],                              #green 
    'G3': [4, 10, 15, 21],                              #green
    #'G4': [4, 10, 5],                                  #green 
    #'G5': [26, 21, 10, 15, 21],                        #green 
    
    'LB1': [23, 14, 17, 5, 15, 16,6,16],              #Lblue 
    'LB2': [25, 15, 12, 7, 19, 21],                     #Lblue 
    #'LB3': [23, 14, 17, 5, 15, 12, 7, 19, 21],         #Lblue 
    #'LB4': [25, 15, 16],                        #Lblue 
    
    'DB1': [20, 12, 21],                                #Dblue Sure
    'DB2': [20, 16, 9,16],                              #Dblue 
    'DB3': [17,16,21],                                  #Dblue 
   
    'Y1': [2, 3, 10],                           #Yellow 
    
    'Y3':  [16,16,8,9,10,10,17],                               #Yellow 
    'Y4' : [15,16,21,21,16,22],
    
    'R1': [17,14,16,12,15,12,7,13,11,16,12,20]          #Red
    
}

'''

#Dblue
db1 = (f'{l[17]}{l[16]}{l[21]}{l[12]}{l[20]}{l[16]}{l[9]}{l[16]}')

#red
r1 = (f'{l[17]}{l[14]}{l[16]}{l[12]}{l[7]}{l[13]}{l[11]}{l[16]}{l[12]}{l[15]}{l[12]}{l[20]}')
r2 = (f'{l[17]}{l[14]}{l[16]}{l[12]}{l[15]}{l[12]}{l[7]}{l[13]}{l[11]}{l[16]}{l[12]}{l[20]}')

#yellow
y1 = (f'{l[2]}{l[3]}{l[10]}{l[10]}{l[17]}')
y2 = (f'{l[16]}{l[16]}{l[8]}{l[9]}{l[10]}{l[10]}{l[17]}')
y3 = (f'{l[15]}{l[16]}{l[21]}{l[21]}{l[16]}{l[22]}')
'''

# Parameters for simulated annealing
initial_temperature = 1.0
cooling_rate = 0.01
iterations = 100000

# Solve the puzzle using simulated annealing with constrained letters
all_solutions, max_score = simulated_annealing(initial_state, puzzle, initial_temperature, cooling_rate, iterations, constrained_letters)

# Initialize a set to keep track of unique second elements
unique_words_set = set()

# Create a new list with unique second elements
filtered_solutions = []

for solution, words in all_solutions:
    unique_words = tuple(words.values())

    if unique_words not in unique_words_set:
        # Add the current tuple to the filtered list
        filtered_solutions.append((solution, words))

        # Add the unique second element to the set
        unique_words_set.add(unique_words)

# Update the original list with the filtered list
all_solutions = filtered_solutions

# Print the results
if max_score >= len(puzzle)-1:
    for solution, words in all_solutions:
        print("Solution:", ''.join(solution))
        print("Words:", words)
    print("All Solutions with Max Score:", max_score)
    print("amount of words", len(puzzle))
else:
    print("Only", max_score, "correct words")