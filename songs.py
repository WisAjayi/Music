import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

# Set up the Spotify API client
client_id = '221bd75853fe4090b932f2bb76d2c18d'
client_secret = '645eb392f93d4e49a05165902c17167e'
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def create_directory():

    base_path = os.getcwd()
    print(base_path)

    new_directory = f"{base_path}/{artist_name}/{album_name}"
    print(new_directory)

    try:
        os.mkdir(new_directory)
        print("Directory created successfully!")
    except FileExistsError:
        print("Directory already exists!")


def save(CLEAN):
    with open(f"{artist_name}/{album_name}/Tracks.txt","a") as G:
        for k in CLEAN:
            G.write(k['name'] + "\n")

# Get artist's albums
artist_name = 'Outkast'
results = sp.search(q='artist:' + artist_name, type='artist')
if results['artists']['items']:
    artist_id = results['artists']['items'][0]['id']
    albums = sp.artist_albums(artist_id, album_type='album')
    for album in albums['items']:
        album_name = album['name']
        album_id = album['id']
        print('Album Name:', album_name)

        create_directory()

        # Get album's tracks
        tracks = sp.album_tracks(album_id)

        save(tracks['items'])


        for track in tracks['items']:
            track_name = track['name']
            track_number = track['track_number']
            print('Track {}: {}'.format(track_number, track_name))
else:
    print('Artist not found.')