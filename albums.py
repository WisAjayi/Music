import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os



def create_directory():

    base_path = os.getcwd()
    print(base_path)

    new_directory = f"{base_path}/{artist_name}"
    print(new_directory)

    try:
        os.mkdir(new_directory)
        print("Directory created successfully!")
    except FileExistsError:
        print("Directory already exists!")






def save(CLEAN):
    with open(f"{artist_name}/Discography.txt","a") as G:
        for k in CLEAN:
            G.write(k['name'] + "\n")




# Set up the Spotify API client
client_id = '221bd75853fe4090b932f2bb76d2c18d'
client_secret = '645eb392f93d4e49a05165902c17167e'
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Get artist information
artist_name = 'Kendrick Lamar'
results = sp.search(q='artist:' + artist_name, type='artist')
artist_info = results['artists']['items'][0]
print('Artist Name:', artist_info['name'])
print('Genres:', artist_info['genres'])

# Get artist discography
artist_id = artist_info['id']
albums = sp.artist_albums(artist_id, album_type='album')


create_directory()
save(albums['items'])




for album in albums['items']:
    print('Album Name:', album['name'])