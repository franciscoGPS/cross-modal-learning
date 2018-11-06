# cross-modal-learning
Dissertation project.


1.- Obtain songs info based on moods

https://developer.spotify.com/dashboard/applications

create an app and get client_id and client_secret.

Client id and secret must be set in spotipy_oauth_lyrics_query_script.py
Setup client_id and client secret in spotipy_oauth_lyrics_query_script.py LINE 63-64

Execute:
 python spotipy_oauth_lyrics_query_script.py -o "example_output" -t 2 -p 1 

-p is the number of playlist per mood to select
-t is the number of tracks per playlist to select

This will generate a csv file of playlists and a csv file of songs

2.- Obtain actual lyrics and audio files

client_secret.json must be set before executing video and lyric retrieval
https://developers.google.com/youtube/registering_an_application?hl=es-419

OAuth client id credentials must be created and downloaded as json.
The json file must be in the same directory

python video_lyric_retrieval.py -o "integral_audios_dataset" 


This will download audio files in mp3 format into the songs directory.

3.- Verify songs were downloaded and compiled into one csv file. 

To cut and sync files with lyrics, the run_dataset_cleanup.py script must be executed.
This will move, convert and cut mp3 files into .wav files. This format is needed to obtain the gammatonegrams.


4.- The network training
For this, the script keras_build.py is executed in the next way:

python keras_build.py -m 20 -n "CNN" -e 25

-m is for the amount of moods (selected random from the list of 20) to consider to make the dataset from.
-e epochs to attempt for the training.
-n network to use. This must be always set to "CNN". It was used to change the type of network to use, but CNN was the only supported at the end of the project.






