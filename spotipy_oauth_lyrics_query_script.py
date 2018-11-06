# https://github.com/plamere/spotipy/blob/master/spotipy/util.py
# http://www.acmesystems.it/python_httpd

from bottle import route, run, request
from bottle import Bottle, template
import spotipy
from spotipy import oauth2
import os
import os.path as path
import sys
import nltk
import re
import sys, getopt
import ipdb


SONGS_FILE = "songs_list"
PLAYLIST_LIMIT = 0
TRACKS_LIMIT= 0
    
def main(argv):

    try:
      global SONGS_FILE, PLAYLIST_LIMIT, TRACKS_LIMIT
      opts, args = getopt.getopt(argv,"hi:o:p:t:",["outputfile=","tracks=", "playlists="])
    except getopt.GetoptError:
      print('spotipy_oauth_lyrics_query_script -o <outputfile> -p <playlists> -t <tracks>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('spotipy_oauth_lyrics_query_script -o <outputfile> -p <playlists> -t <tracks>')
         sys.exit()
      elif opt in ("-t", "--tracks"):
         TRACKS_LIMIT = arg.strip()
      elif opt in ("-p", "--playlists"):
         PLAYLIST_LIMIT = arg.strip()
      elif opt in ("-o", "--outputfile"):
         SONGS_FILE = arg.strip()

    print( 'Output file is: ', SONGS_FILE)
    print( 'Playlists per query: ', PLAYLIST_LIMIT)
    print( 'Tracs per playlist: ', TRACKS_LIMIT)
   

#filesystem_path = (path.abspath(path.join(path.dirname("__file__"), '../../')) + '/youtdl/')

#sys.path.append(filesystem_path)

import make_table as table 

PORT_NUMBER = 8080
SPOTIPY_CLIENT_ID = 'c2100406ea754d2b87c9a5118d52efe8'
SPOTIPY_CLIENT_SECRET = 'd653ca6c7c564a7f975103cbfc3e58e7'
SPOTIPY_REDIRECT_URI = 'http://localhost:8080/'
SCOPE = 'user-library-read'
CACHE = '.spotipyoauthcache'

MOODS = ["aggressive", "angry", "bittersweet", "calm", "depressing", "dreamy", "fun", "gay", "happy", "heavy", "intense", "melancholy", "playful", "quiet", "quirky", "sad", "sentimental", "sleepy", "soothing", "sweet"]
OPEN_MODE_WRITE = "w+"
OPEN_MODE_APPEND = "a"
PLAYLISTS_FILE = "playlyst_dictionary"
##SONGS_FILE = "moods_songs_clear_names_file"
#PLAYLIST_LIMIT = 25
#TRACKS_LIMIT= 25



sp_oauth = oauth2.SpotifyOAuth( SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET,SPOTIPY_REDIRECT_URI,scope=SCOPE,cache_path=CACHE )
bootle = Bottle()
@route('/')
def index():
    access_token = ""
    filename = "index.html"
    root = os.path.join(os.path.dirname(__file__), 'status', 'public_html')

    token_info = sp_oauth.get_cached_token()

    if token_info:
        print ("Found cached token!")
        access_token = token_info['access_token']
    else:
        url = request.url
        code = sp_oauth.parse_response_code(url)
        if code:
            print ("Found Spotify auth code in Request URL! Trying to get valid access token...")
            token_info = sp_oauth.get_access_token(code)
            access_token = token_info['access_token']

    if access_token:
        print ("Access token available! Trying to get user information...")
        client = spotipy.Spotify(access_token)
        results = client.current_user()
        
        procesed_playlists = make_calls_for_playlists(client)
        
        if save_as_table(procesed_playlists, PLAYLISTS_FILE, OPEN_MODE_WRITE) == 1:
            
            processed_tracks = get_songs_from_playlists(client, procesed_playlists)

            if save_as_table(processed_tracks, SONGS_FILE, OPEN_MODE_APPEND) == 1:

            #results = sp.search(q=mood, limit=10, type='playlist')

                info={'result': "true"}
            else:
                info={'result': "false"}
            
        
        return template('views/template.tpl', info)

        
        
        #return bottle.static_file(filename, root=root)
        

    else:
        return htmlForLoginButton()
        exit()



def clean_string(string_to_clean):
    pattern = re.compile(r"(instrumental|acoustic|radio|edit|version|remix|MTV|Unplugged)\W|[(].*[)]|[\◐\’\'\-\_\$\@\%\&\!\?\,\.\[\]\*\+\/\#\`\~\^\(\)\_\\\"\:\>\<\;]\s*", re.I)
    return pattern.sub(" ",string_to_clean ).strip()



def get_songs_from_playlists(sp,playlists):
    tracks = list()
    song_lyrics_list = {'mood':[], 'title':[], 'artist':[], 'lyric':[], 'youtube_video_id':[], 'file':[], 'bow':[]}
    #moods_list_indexes_retrieved = {'aggressive': 0,'angry': 0, 'bittersweet': 0, 'calm': 0, 'depressing': 0, 'dreamy': 0, 'fun': 0, 'gay': 0, 'happy': 0, 'heavy': 0, 'intense': 0, 'melancholy': 0, 'playful': 0, 'quiet': 0, 'quirky': 0, 'sad': 0, 'sentimental': 0, 'sleepy': 0, 'soothing': 0, 'sweet':0}
    moods_list_indexes_retrieved = { i : 0 for i in MOODS }
    traks_sum = 0
    total_tracks = 0;
    mood = ''

    for item in playlists:
        row = item.split(',')
        #results = sp.search(q=row[], limit=10, type='playlist')

        username = row[2].strip()
        playlist_id = row[1].strip()
        #bp.set_trace()
        
        if mood == "" or mood != row[0]:
            print("", traks_sum, mood)
            mood = row[0]
            total_tracks += traks_sum
            traks_sum = 0
        
        try:
            pass
            playlists = sp.user_playlist_tracks(username, playlist_id, limit=TRACKS_LIMIT)
            track_sp_id = set()
            for item in playlists['items']:
                
                track_id = item['track']['id']
                before_len = len(track_sp_id)
                track_sp_id.add(track_id)

                if len(track_sp_id) > before_len:

                    duration = item['track']["duration_ms"]/1000/60
                    if duration > 0.5 or duration < 7:

                        track_name = clean_string(item['track']['name'])
                        artist_name = clean_string(item['track']['artists'][0]['name'])
                        


                        song = row[0] + ", "+ track_name + ", "+ artist_name+ ","+ track_id 
                        moods_list_indexes_retrieved[row[0]]+=1                        
                        #for artist_item in item['artist']
                            #artist_name += artist_item['name'] + " "
                        tracks.append(song)
                        traks_sum += 1
        except Exception as e:
            print(e)
            print(row)
        
    
    print("", traks_sum, mood)
    print("", total_tracks)    
    print(moods_list_indexes_retrieved)
    return tracks


def htmlForLoginButton():
    auth_url = getSPOauthURI()
    htmlLoginButton = "<a href='" + auth_url + "'>Login to Spotify</a>"
    return htmlLoginButton

def getSPOauthURI():
    auth_url = sp_oauth.get_authorize_url()
    return auth_url



def make_calls_for_playlists(client):
    playlists = list()
    sum_tracks = 0
    sum_moods = 0

    for j, mood in enumerate(MOODS):
        """
        if j == 8:
            PLAYLIST_LIMIT = 20
        else:
            PLAYLIST_LIMIT = 15
        """
        results = client.search(q=mood, limit=PLAYLIST_LIMIT, type='playlist')
        for i, t in enumerate(results['playlists']['items']):
            
            """
            print(t['owner']['id'])
            print(t['owner']['uri'])
            print(t['name'])
            print(t['tracks']['total'])
            print(t['owner']['id'])
            print()
            """
            try:
                pass
                sum_tracks += t['tracks']['total']
            except Exception as e:
                print(t)
            playlists.append(mood + ", "+ t['id'] + ", "+t['owner']['id'] + ", "+ t['owner']['uri'] + ", "+ t['name'] + ", "+ str(t['tracks']['total']) +"" ) 
        print ("total in : ",mood ,sum_tracks)
        
        
        sum_moods += sum_tracks
        sum_tracks = 0
    print("Total songs of all moods: ", sum_moods)
    return playlists

#This method receives an array of playists as csv and stores it.
def save_as_table(playlists, filename, open_mode):
    table.make_ids_table(filename, playlists, open_mode)    
    return 1


def eng_ratio(text):
    ''' Returns the ratio of non-English to English words from a text '''

    english_vocab = set(w.lower() for w in nltk.corpus.words.words()) 
    text_vocab = set(w.lower() for w in text.split() if w.lower().isalpha()) 
    unusual = text_vocab.difference(english_vocab)
    diff = 1
    if len(text_vocab) > 0:
        diff = len(unusual)/len(text_vocab)
    return diff
    

if __name__ == '__main__':
    main(sys.argv[1:])
    run(host='localhost', port=8080)
