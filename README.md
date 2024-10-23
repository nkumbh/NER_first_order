# NER_first_order

* Problem Statement:*
● The assignment targets to implement Hidden Markov Model (HMM) to perform
Named Entity Recognition (NER) task
Implementation:
HMM based Model:
● HMM Parameter Estimation
○ Input: Annotated tagged dataset
○ Output: HMM parameters
○ Procedure:
■ Step1: Find states.
■ Step2: Calculate Start probability (π).
■ Step3: Calculate transition probability (A)
■ Step4: Calculate emission probability (B)
● Features for HMM:
○ Train two HMM models based on:
■ First order markov assumption (Bigram) where current word NER
tag is based on the previous and current words
■ Second order markov assumption (Trigram) where current word
NER tag is based on the current word along with the previous two
words
RNN based Model:
● Explain and draw the architecture of RNN that you are proposing with justification
● Describe the features of RNN
Testing:
● After calculating all these parameters apply these parameters to the Viterbi
algorithm and test sentences as an observation to find named entities
Dataset:
● Dataset consists of tweets and each word is tagged with its corresponding NER
tag
● NER-Dataset-Train.txt —> Contains train set
● Tweet NER dataset: Link to dataset
● Format of dataset:
○ Each line contains <Word \t Tag> (word followed by tab-space and tag)
○ Sentences are separated by a new line
**Imports**
import pandas as pd

# Read the text file
with open('NER-Dataset-Train.txt', 'r') as file:
    data = file.readlines()
# Concatenate the lines to form sentences
sentences = ' '.join(data).split('\n')

# Split each sentence into words and tags
word_tag_pairs = [sentence.strip().split() for sentence in sentences if sentence.strip()]
word_tag_pairs[:5]
# Create separate lists for words and tags
df = pd.DataFrame(word_tag_pairs, columns=['Word', 'Tag'])
df.head()
In this notebook we will look at the NER dataset and use it to understand HMM
# Step 1: Find states

states = df['Tag'].unique()
print(states)
# Step 2: Calculate Start Probability (π)
start_counts = df.groupby('Tag').size().div(len(df))
start_prob = start_counts.to_dict()
start_prob
# Step 3: Calculate Transition Probability (A)

import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, Counter

transition_counts = defaultdict(lambda: defaultdict(int))
for i in range(len(df) - 1):
    current_tag = df.at[i, 'Tag']
    next_tag = df.at[i + 1, 'Tag']
    transition_counts[current_tag][next_tag] += 1

transition_prob = {tag: {next_tag: count / sum(next_tag_counts.values())
                         for next_tag, count in next_tag_counts.items()}
                   for tag, next_tag_counts in transition_counts.items()}

# Encode tags
le = LabelEncoder()
df['Tag'] = le.fit_transform(df['Tag'])
df.head()
transition_prob
# Function to calculate emission probabilities

emission_counts = defaultdict(Counter)
for i in range(len(df)):
    tag = df.loc[i, 'Tag']
    word = df.loc[i, 'Word']
    emission_counts[tag][word] += 1

emission_prob = {tag: {word: count / sum(words.values()) for word, count in words.items()}
                 for tag, words in emission_counts.items()}

# Define the HMM Model:
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# States (Tags)
states = list(le.classes_)
n_states = len(states)

# Vocabulary
words = df['Word'].unique()
n_words = len(words)
# Viterbi Algorithm for decoding
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p.get(y, 0) * emit_p[y].get(obs[0], 0)
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max((V[t-1][y0] * trans_p[y0].get(y, 0) * emit_p[y].get(obs[t], 0), y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        path = newpath
   
    # Return the most probable sequence
    n = len(obs) - 1
    (prob, state) = max((V[n][y], y) for y in states)
    return path[state]


# Function to convert tags to state names
def decode_states(encoded_states, le):
    return [le.inverse_transform([state])[0] for state in encoded_states]

# Prepare sequences
words = df['Word'].values
tags = df['Tag'].values

# Perform 5-fold cross-validation
kf = KFold(n_splits=5)

accuracies = []
precisions = []
recalls = []
f_scores = []

for train_index, test_index in kf.split(df):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
            
    test_words = test_df['Word'].values
    test_tags = test_df['Tag'].values
    
    predicted_tags = []
    for word in test_words:
        observed = [word]
        states_seq = viterbi(observed, range(n_states), start_prob, transition_prob, emission_prob)
        predicted_tags.append(states_seq[0])
    
    accuracy = accuracy_score(test_tags, predicted_tags)
    precision, recall, f_score, _ = precision_recall_fscore_support(test_tags, predicted_tags, average='weighted')
    
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f_scores.append(f_score)

print("Cross-Validation Results:")
for i in range(5):
    print(f"Fold {i+1} - Accuracy: {accuracies[i]:.4f}, Precision: {precisions[i]:.4f}, Recall: {recalls[i]:.4f}, F-Score: {f_scores[i]:.4f}")

print("\nAverage Results:")
print(f"Accuracy: {np.mean(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f}")
print(f"F-Score: {np.mean(f_scores):.4f}")
test_sentences = [
    "Narendra lives in Pune.",
    "IIT Patna. is a major tech college.",
    "The Eiffel Tower is in Paris."
]

# Tokenize the sentences (assuming simple space-based tokenization for this example)
test_sentences = [sentence.split() for sentence in test_sentences]

# Get unique states (tags) from the label encoder
states = list(range(len(le.classes_)))
states
# Predict tags for each test sentence using model
for sentence in test_sentences:
        
    # Second-order (trigram) HMM
    first_order_tags = viterbi(sentence, states, start_prob, transition_prob, emission_prob)
    first_order_tags_decoded = decode_states(first_order_tags, le)
    first_order_tags_decoded_ = []
    for i in first_order_tags_decoded:
        first_order_tags_decoded_.append(str(i))
    print(f"Sentence: {' '.join(sentence)}")
    print(f"Second-Order HMM Tags: {' '.join(first_order_tags_decoded_)}")
    print()
