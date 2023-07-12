import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import os

### Use to get Lyrics and create Artist file and album files ###

# Set up the Spotify API client
client_id = '' ### REPLACE WITH YOUR SPOTIFY client_id api key  or add to env and use os.environ.get('') ###
client_secret = '' ### REPLACE WITH YOUR SPOTIFY client_secret api key  or add to env and use os.environ.get('') ###
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Set up the Genius API client
access_token = '' ### REPLACE WITH YOUR GENIUS access_token api key  or add to env and use os.environ.get('') ###
genius = lyricsgenius.Genius(access_token)

### Create Album & Track.txt file in order to save lyrics ###
artist_name = 'Dj Quik'
album_name = 'Safe + Sound'
track_name = 'Dollaz + Sense'


def save(i):
    with open(f"Artists/{artist_name}/{album_name}/{track_name}.txt","a") as G:
        G.write(i)



def create_directory():

    base_path = os.getcwd() # Current Directory #
    print(base_path)

    new_directory = f"{base_path}/Artists/{artist_name}"
    print(new_directory)

    try:
        os.mkdir(new_directory)
        print("Directory created successfully!")
    except FileExistsError:
        print("Directory already exists!")
        
        
    
    final_directory = f"{new_directory}/{album_name}"
    try:
        os.mkdir(final_directory)
       
    except FileExistsError:
        print("Directory already exists!")



# Get track information from Spotify API
def lyrics():

    results = sp.search(q='artist:' + artist_name + ' track:' + track_name, type='track')
    if results['tracks']['items']:
        track_info = results['tracks']['items'][0]
        print('Artist Name:', track_info['artists'][0]['name'])
        print('Track Name:', track_info['name'])

        # Get song lyrics from Genius API
        song = genius.search_song(track_info['name'], track_info['artists'][0]['name'])
        if song:
            print('Song Lyrics:', song.lyrics)
            save(song.lyrics) # Saves To File.
    else:
        print('Track not found.')




def group_create():
    
    with open(f"{artist_name}/{album_name}/Tracks.txt","r") as file:
        all = file.readlines()
        print(all)


    for i in all:
        track_name = i.strip()
        print(track_name)
        lyrics()
        

create_directory()      
lyrics()