import time
import sys
from os import walk
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

EXAMPLE_FILE = './Artists/Kanye West/Amazing.txt'

ALBUM_1_PATH = './Artists/Kanye West/808s & Heartbreak'
ALBUM_2_PATH = './Artists/Kanye West/Donda'
CLASSIFIED_1_PATH = './Artists/Kanye West/00_CLASSIFIED'
CLASSIFIED_2_PATH = './Artists/Kanye West/01_CLASSIFIED'

ALBUM_CAT = 1
CLASSIFIED_CAT = 0
VOCAB_SIZE = 464

DATA_JSON_FILE = './Artists/Kanye West/00_song-data.json'
WORD_ID_FILE = './Artists/Kanye West/01_word-by-id.csv'

TRAINING_DATA_FILE = './Artists/Kanye West/02_training.txt'
TEST_DATA_FILE = './Artists/Kanye West/03_testing.txt'


stream = open(EXAMPLE_FILE, encoding='utf-8')
message = stream.read()
stream.close()

print("File Encoding = ", sys.getfilesystemencoding() )

stream = open(EXAMPLE_FILE, encoding='utf-8')
is_body = False
lines = []
for line in stream:
    if is_body:
        lines.append(line)
    elif line == '\n':
        is_body = True
stream.close()
song_body = '\n'.join(lines)



def generate_song_body(path):
    
    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            
            filepath = join(root, file_name)
            
            stream = open(filepath, encoding='utf-8')

            is_body = False
            lines = []

            for line in stream:
                if is_body:
                    lines.append(line)
                elif line == '\n':
                    is_body = True

            stream.close()

            song_body = '\n'.join(lines)
            
            yield file_name, song_body
            


          
def df_from_directory(path, classification):
    
    rows = []
    row_names = []
    
    for file_name, song in generate_song_body(path):
        rows.append({'SONG_LYRICS': song, 'CATEGORY': classification}) # Category number added Here #
        row_names.append(file_name)
        
    return pd.DataFrame(rows, index=row_names)



all_songs_in_album = df_from_directory(ALBUM_1_PATH, 1)
#all_songs_in_album = all_songs_in_album._append(df_from_directory(ALBUM_2_PATH, 1)) ### CHANGE HERE TO REMOVE 2ND ALBUM ###
#print(all_songs_in_album.head())
#print(all_songs_in_album.tail())
print(all_songs_in_album.shape)


all_classified_songs = df_from_directory(CLASSIFIED_1_PATH, CLASSIFIED_CAT)
all_classified_songs = all_classified_songs._append(df_from_directory(CLASSIFIED_2_PATH, CLASSIFIED_CAT))
print(all_classified_songs.shape)


data = pd.concat([all_songs_in_album, all_classified_songs])
print('Shape of entire dataframe is ', data.shape)
#print(data.head())
#print(data.tail())


print( data['SONG_LYRICS'].isnull().values.any() )
# check if there are empty songs (string length zero)
print( (data.SONG_LYRICS.str.len() == 0).any() )
print( (data.SONG_LYRICS.str.len() == 0).sum() )
# Checking the number of entries with null/None values?
print( data.SONG_LYRICS.isnull().sum() )


# # # # LOCATE EMPTY EMAIL # # # #
print( type(data.SONG_LYRICS.str.len() == 0) )
print( data[data.SONG_LYRICS.str.len() == 0].index )

# # # # REMOVE SYSTEM FILE ENTRY FROM DATAFRAME # # # #
data.drop(['Tracks.txt'], inplace=True)
print("Shape of total data = ", data.shape )



# # # # Add Documents ID to track Emails in Dataset
document_ids = range(0, len(data.index))
data['DOC_ID'] = document_ids
data['FILE_NAME'] = data.index
data.set_index('DOC_ID', inplace=True)
print ( data.head() )
print( data.tail() )
data.to_json(DATA_JSON_FILE)


###   NATURAL   LANGUAGE   PROCESSING     ###
#nltk.download()

msg = 'All work and no play makes Jack a dull boy.'
word_tokenize(msg.lower())

stop_words = set(stopwords.words('english'))
print( type(stop_words) ) # set


if 'this' in stop_words: print('Found it!')
if 'hello' not in stop_words: print('Nope. Not in here')

msg = 'All work and no play makes Jack a dull boy. To be or not to be.'
words = word_tokenize(msg.lower())
filtered_words = []
# append non-stop words to filtered_words
for word in words:
    if word not in stop_words:
        filtered_words.append(word)

print(filtered_words)




# # # word stemming # # #
msg = 'All work and no play makes Jack a dull boy. To be or not to be. Nobody expects the Spanish Inquisition!'
words = word_tokenize(msg.lower())
# stemmer = PorterStemmer()
stemmer = SnowballStemmer('english')


filtered_words = []
# append non-stop words to filtered_words
for word in words:
    if word not in stop_words:
        stemmed_word = stemmer.stem(word)
        filtered_words.append(stemmed_word)

print(filtered_words)


# # # removing punctuation  # # # To be removed. 
print('p'.isalpha())
print('?'.isalpha())
msg = 'All work and no play makes Jack a dull boy. To be or not to be. ??? Nobody expects the Spanish Inquisition!'

words = word_tokenize(msg.lower())
stemmer = SnowballStemmer('english')
filtered_words = []

for word in words:
    if word not in stop_words and word.isalpha():
        stemmed_word = stemmer.stem(word)
        filtered_words.append(stemmed_word)

#print(filtered_words)


def clean_song(song_name, stemmer=PorterStemmer(), 
                 stop_words=set(stopwords.words('english'))):
    
    # Converts to Lower Case and split up the words.
    words = word_tokenize(song_name.lower())
    
    filtered_words = []
    
    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
            
    #print(filtered_words)
    return filtered_words

clean_song(song_body)

#print( data.iat[2, 2] )
#print( data.iloc[5:11] )

first_songs = data.SONG_LYRICS.iloc[0:3] ### A TEST NOT RELEVANT ###
nested_list = first_songs.apply(clean_song)
flat_list = [item for sublist in nested_list for item in sublist]       
#print( len(flat_list) )
#print( flat_list )



# # # # Using Logic to Slice Dataframes # # # #
print("Shape for CATEGORY = ", data[data.CATEGORY == 1].shape )
print( data[data.CATEGORY == 1].tail() )

doc_ids_song = data[data.CATEGORY == 1].index
doc_ids_classified = data[data.CATEGORY == 0].index
print( doc_ids_classified )


nested_list_classified = nested_list.loc[doc_ids_classified]
print( "Shape of nested list = ", nested_list_classified.shape )
print( nested_list_classified.tail() )



flat_list_classified = [item for sublist in nested_list_classified for item in sublist]
normal_words = pd.Series(flat_list_classified).value_counts()
print("Shape of normal words = ",normal_words.shape[0]) # total number of unique words in the classified songs
print(normal_words[:10])



def nested_list():
    
    nested_list_song = nested_list.loc[doc_ids_song]
    flat_list_song = [item for sublist in nested_list_song for item in sublist]
    song_words = pd.Series(flat_list_song).value_counts()
    print(song_words.shape[0]) # total number of unique words in the spam messages
    print( song_words[:10] )
    
    
    
# # # GENERATE VOCABULARY & DICTIONARY # # #
stemmed_nested_list = data.SONG_LYRICS.apply(clean_song)
flat_stemmed_list = [item for sublist in stemmed_nested_list for item in sublist]
unique_words = pd.Series(flat_stemmed_list).value_counts()
print('Number of unique words', unique_words.shape[0])
print( unique_words.head() )

frequent_words = unique_words[0:VOCAB_SIZE]
print('Most common words: \n', frequent_words[:10])


## Create Vocabulary DataFrame with a WORD_ID ##
word_ids = list(range(0, VOCAB_SIZE))
vocab = pd.DataFrame({'VOCAB_WORD': frequent_words.index.values}, index=word_ids)
vocab.index.name = 'WORD_ID'
print( vocab.head() )


### SAVE THE VOCABULARY TO A CSV FILE ###
vocab.to_csv(WORD_ID_FILE, index_label=vocab.index.name, header=vocab.VOCAB_WORD.name)

if 'kim' in set(vocab.VOCAB_WORD): print("yessir")


clean_song_lengths = []
for sublist in stemmed_nested_list:
    clean_song_lengths.append(len(sublist))
    
clean_song_lengths = [len(sublist) for sublist in stemmed_nested_list]
print('lyric of the longest song in the album:', max(clean_song_lengths))
print('Song position in the list (and the data dataframe)', np.argmax(clean_song_lengths))

print( stemmed_nested_list[np.argmax(clean_song_lengths)] )
print( data.at[np.argmax(clean_song_lengths), 'SONG_LYRICS'] )



# Generate Features & a Sparse Matrix
### Creating a DataFrame with one Word per Column
print( type(stemmed_nested_list) )
print( type(stemmed_nested_list.tolist()) )
word_columns_df = pd.DataFrame.from_records(stemmed_nested_list.tolist())
print( word_columns_df.head() )
print( word_columns_df.shape )

X_train, X_test, y_train, y_test = train_test_split(word_columns_df, data.CATEGORY,test_size=0.4, random_state=50)

print('Nr of training samples', X_train.shape[0])
print('Fraction of training set', X_train.shape[0] / word_columns_df.shape[0])

X_train.index.name = X_test.index.name = 'DOC_ID'
print( X_train.head() )
print("y_train = ",y_train.head() )

### Create a Sparse Matrix for the Training Data ###
word_index = pd.Index(vocab.VOCAB_WORD)
print( type(word_index[3]) )
print( word_index.get_loc('heartbreak') )


def make_sparse_matrix(df, indexed_words, labels):
    """
    Returns sparse matrix as dataframe.
    
    df: A dataframe with words in the columns with a document id as an index (X_train or X_test)
    indexed_words: index of words ordered by word id
    labels: category as a series (y_train or y_test)
    """
    
    nr_rows = df.shape[0]
    nr_cols = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []
    
    for i in range(nr_rows):
        for j in range(nr_cols):
            
            word = df.iat[i, j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = indexed_words.get_loc(word)
                category = labels.at[doc_id]
                
                item = {'LABEL': category, 'DOC_ID': doc_id,
                       'OCCURENCE': 1, 'WORD_ID': word_id}
                
                dict_list.append(item)
    
    return pd.DataFrame(dict_list)


sparse_train_df = make_sparse_matrix(X_train, word_index, y_train)

print( sparse_train_df[:5] )
print( sparse_train_df.shape )
print( sparse_train_df[-5:] )


### Combine Occurrences with the Pandas groupby() Method ###
train_grouped = sparse_train_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()
print( train_grouped.head() )
print( vocab.at[0, 'VOCAB_WORD'] ) ### Can be Changed ###

print(data)
print( data.SONG_LYRICS[0] ) ### Can be changed to print different lyrics from album ###

train_grouped = train_grouped.reset_index()
print( train_grouped.head() )
print( train_grouped.tail() )
print( vocab.at[460, 'VOCAB_WORD'] )
print( data.SONG_LYRICS[10] ) ### Can be changed to print different lyrics from album ###
print( train_grouped.shape )

### Save Training Data as .txt File ###
np.savetxt(TRAINING_DATA_FILE, train_grouped, fmt='%d')
print( train_grouped.columns )


### Group the occurrences of the same word in the same email. Then save the data as a .txt file.  ###
print( X_test.head() )
print( y_test.head() )
print( X_test.shape )

sparse_test_df = make_sparse_matrix(X_test, word_index, y_test)
print( sparse_test_df.shape )
test_grouped = sparse_test_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum().reset_index()
print( test_grouped.head() )
print( test_grouped.shape )
np.savetxt(TEST_DATA_FILE, test_grouped, fmt='%d')




def get_data():
    
    return data