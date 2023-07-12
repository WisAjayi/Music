from wordcloud import WordCloud
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk


EXAMPLE_FILE = './Artists/Kanye West/Amazing.txt'

WHALE_FILE = './IMG/whale-icon.png'
SKULL_FILE = './IMG/skull-icon.png'
THUMBS_UP_FILE = './IMG/thumbs-up.png'
THUMBS_DOWN_FILE = './IMG/thumbs-down.png'
STAR_FILE = './IMG/star.png'
LOCATION_FILE = './IMG/loc.png'
COMMENT_FILE = './IMG/comment.png'
UPVOTE_FILE = './IMG/upvote.png'
USER_FILE = './IMG/user.png'
PRINCESS_FILE = './IMG/princess.JPG'
CUSTOM_FONT_FILE = './Fonts/OpenSansCondensed-Bold.ttf'

    
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

 


def example_wordcloud():
    
    word_cloud = WordCloud().generate(song_body)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()





def moby_dick_wordcloud():

    example_corpus = nltk.corpus.gutenberg.words('melville-moby_dick.txt')
    print( len(example_corpus) )
    print( type(example_corpus) )
    print( example_corpus )
    
    hamlet_corpus = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
        
    word_list = [''.join(word) for word in example_corpus]
    novel_as_string = ' '.join(word_list)

    icon = Image.open(WHALE_FILE)
    image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
    image_mask.paste(icon, box=icon)
    rgb_array = np.array(image_mask) # converts the image object to an array

    word_cloud = WordCloud(mask=rgb_array, background_color='white', max_words=400, colormap='ocean')
    word_cloud.generate(novel_as_string)

    plt.figure(figsize=[16, 8])
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    print( rgb_array.shape )
    print( rgb_array[1023, 2047] )
    print( rgb_array[500, 1000] )


def create_wordcloud(Filename,filetxt,font=None,fontsize=None,cm=None):



    icon = Image.open(Filename)
    image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
    image_mask.paste(icon, box=icon)
    rgb_array = np.array(image_mask)

    word_cloud = WordCloud(mask=rgb_array, background_color='white',colormap=cm, max_words=600,font_path=font,max_font_size=fontsize) # max_font_size=2000, font_path=CUSTOM_FONT_FILE
    word_cloud.generate(filetxt)
    plt.figure(figsize=[16, 8])
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()



create_wordcloud(WHALE_FILE,song_body)
create_wordcloud(THUMBS_UP_FILE,song_body,font=CUSTOM_FONT_FILE,fontsize=2000,cm='gist_heat')
create_wordcloud(SKULL_FILE,song_body)
