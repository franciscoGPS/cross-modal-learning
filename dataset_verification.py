import ipdb
import pandas as pd
import os
import os.path as path
import sys
from shutil import copyfile
import make_lyrics_table as mlt
import time
import datetime
import nltk
import re
from shutil import move
import glob
import subprocess
import re
from shlex import quote

import numpy as np

import scipy.io.wavfile as wav

from gammatone import gtgram


filesystem_path = (path.abspath(path.dirname("__file__")))


sys.path.append(filesystem_path)

from filesystem_tools import Filesystem
from youtube_query_for_video_ids import QueryVideo
from song import Song
import youtube_dl_client as ytc
#audios_path = "/media/frisco/FAT/cross-modal/audios/"

MOODS = ['angry', 'aggressive', 'bittersweet', 'calm', 'depressing', 'dreamy', 'fun', 'gay', 'happy', 'heavy', 'intense', 'melancholy', 'playful', 'quiet', 'quirky', 'sad', 'sentimental', 'sleepy', 'soothing', 'sweet']


def make_audio_table(audios_path):
	
	files = [os.path.join(audios_path,fn) for fn in os.listdir(audios_path) if fn.endswith('.mp3')]
	data = {'mood':[], 'title':[], 'artist':[], 'lyric':[], 'youtube_video_id':[], 'file':[]}
	# Add artist and title data to dictionary
	if len(files) > 0:
		for f in files:
			data['file'].append(f)
			audio_name = f.split('/')[-1]
			audio_data = re.split(r"_",audio_name, 3)	
			data['mood'].append(MOODS.index(audio_data[0])+1)
			data['artist'].append(audio_data[1])
			data['title'].append(audio_data[2])
		
			if len(audio_data) <= 4  :
				data['youtube_video_id'].append(audio_data[3].split(".")[0])
			else:
				data['youtube_video_id'].append(audio_data[-1].split(".")[0])

			data['lyric'] = ""
			# Convert dictionary to pandas DataFrame
			df = pd.DataFrame.from_dict(data, orient='columns')
			df = df[['mood', 'title', 'artist', 'lyric', 'youtube_video_id', 'file' ]]
		
		return df
	else:
		df = pd.DataFrame.from_dict(data, orient='columns')
		return df




def cut_30s_from_file(audio_new_name, file_path, output_path, start_time="00:30"):
	
	#"ffmpeg -i somefile.mp3 -f segment -segment_time 3 -c copy out%03d.mp3"

	subfolder_name = "30s_cuts/"
	if not os.path.isdir(output_path+subfolder_name):
		create_sub_dir = "mkdir "+ output_path+subfolder_name
		subprocess.call(create_sub_dir, shell=True)

	#file_path = escape_especial_chars(file_path)
	basename = os.path.basename(file_path)
	#command = "ffmpeg -i \""+ file_path +"\" -ab 160k -ac 1 -ar 22050 -t 30 -ss 00:00:30.000 -vn "+ output_path+subfolder_name + audio_new_name+ ".wav"
	file_to_cut = quote("/".join(file_path.split("/")[:-1]) +"/" +str(basename))
	command = 'ffmpeg -i '+ file_to_cut +' -ab 160k -ac 1 -ar 22050 -t 30 -ss 00:'+start_time+'.000 -vn "'+ output_path+subfolder_name + audio_new_name+ '".wav'
	result = subprocess.call(command, shell=True)
	if result:
		print(result)


def escape_especial_chars(input_string):
	escaped = input_string.translate(str.maketrans({"$":  r"\$"}))
	return escaped


def cleanup(csv_file, audios_path, delivery_path):
	
	songs_folder_df = make_audio_table(audios_path)
	
	cut = True
	files = [os.path.join(filesystem_path,fn) for fn in os.listdir(filesystem_path) if fn.endswith(csv_file+".csv")]
	#ipdb.set_trace()
	analysis_result = {'found': 0, 'not_found': 0, 'other': 0 }
	print("Reading files found: ", len(files))
	for file in files:
		songs_lyrics_ids_file = pd.read_csv(file, delimiter=',', encoding="utf-8",quotechar='"')
		

		date = str(datetime.datetime.now()).replace(" ", "_" )
		output_file_name = date+"_integral_synced_SongsMoodsFile.csv"
		#ipdb.set_trace()

		song_lyrics_list = {'mood_code':[], 'mood':[], 'title':[], 'artist':[], 'lyric':[], 'youtube_video_id':[], 'file':[], 'bow':[]}
		songs_features = {'mood_code':[],'gtg':[], 'mfcc':[]}
		dataframe = pd.DataFrame.from_dict(song_lyrics_list, orient='columns')

		#dataframe = dataframe[['mood_code', 'mood', 'title', 'artist', 'youtube_video_id', 'file', 'bow','lyric', 'gtg', 'mfcc' ]]
		with open(output_file_name, 'a') as f:
			dataframe.to_csv(f)

		print("This files contains rows", len(songs_lyrics_ids_file))
		for row_id, song in songs_lyrics_ids_file.iterrows():

			video_id = ""
			try:
				video_id = song[5]
			except Exception as e:
				print("error " + song)
				
			#video_id = songs_lyrics_ids_file[row_id]['youtube_video_id']
			
			query_result = songs_folder_df.query('youtube_video_id ==  @video_id')
			if len(query_result) == 1:
				
				print("audio (and lyric) found in CSV")

				audios_found = glob.glob(audios_path+'*_'+video_id+'.mp3')
				if len(audios_found) == 1:
					print(audios_found)
					
					try:
						
						if os.path.exists(str(audios_found[0])):

							mood = song[1].strip()
							title = song[2].strip()
							artist = song[3].strip()
							lyric = song[4].strip()
							youtube_video_id = song[5].strip()
							filepath = song[6].strip()
							bow = song[7].strip()
							


							title = re.sub('\W+','_', title)
							artist = re.sub('\W+','_', artist)
							filename = str(audios_found[0]).split('/')[-1]
							full_delivery_path = delivery_path+filename
							
							#result = copyfile(str(audios_found[0]), full_delivery_path)						
							sample_audio_name_composition = title+"_"+artist
							
							sample_audio_name_composition = mood+"_"+sample_audio_name_composition+"_"+video_id
							
							#cut_30s_from_file(sample_audio_name_composition, full_delivery_path, delivery_path)
							
							new_file_name_path = delivery_path+"30s_cuts/"+sample_audio_name_composition+".wav"
							
							


							saved_files = [os.path.join(delivery_path,fn) for fn in os.listdir(delivery_path) if fn.endswith('.mp3')]
							
							
							if len(song_lyrics_list['file']) <= len(saved_files) or not cut:
								print("song_lyrics_list len: ", len(song_lyrics_list['file']), "saved_files len: ",len(saved_files))
									
								analysis_result['found'] += 1 
								song_lyrics_list['title'].append(title)
								song_lyrics_list['artist'].append(artist)
								song_lyrics_list['mood'].append(mood)
								song_lyrics_list['mood_code'].append(MOODS.index(mood))
								song_lyrics_list['lyric'].append(lyric)
								song_lyrics_list['youtube_video_id'].append(youtube_video_id)
								song_lyrics_list['bow'].append(bow)
								song_lyrics_list['file'].append(delivery_path+query_result['file'].values[0].split('/')[-1])
								
								#songs_features['gtg'].append(gtg)
								#songs_features['mfcc'].append(mfcc)
								#songs_features['mood_code'].append(MOODS.index(mood))
							else:
								ipdb.set_trace()
							#os.rename(query_result['file'].values[0], full_delivery_path)

						#ipdb.set_trace()
					except Exception as e:
						print("  ################################################################################################  ")
						print(e)
						print("  ################################################################################################  ")
						
						#os.rename(delivery_path+query_result['file'].values[0].split('/')[-1], query_result['file'].values[0])
				elif len(audios_found) > 1:
					print("More than one audio file found", audios_found)
					#copyfile(src, dst)
				
			elif len(query_result) == 0:
				#ipdb.set_trace()
				analysis_result['not_found'] += 1
				print("audio not found in sounds folder: ", video_id)
				#songs_lyrics_ids_file.drop(songs_lyrics_ids_file.index[row_id])
				#songs_lyrics_ids_file = songs_lyrics_ids_file[songs_lyrics_ids_file['youtube_video_id'] != video_id]


			else:
				#ipdb.set_trace()
				analysis_result['other'] += 1
				print("more than one result")
				print(query_result)

			dataframe = pd.DataFrame.from_dict(song_lyrics_list, orient='columns')
			with open(output_file_name, 'a') as f:
				dataframe.to_csv(f, header=False)

			song_lyrics_list = {'mood_code':[], 'mood':[], 'title':[], 'artist':[], 'lyric':[], 'youtube_video_id':[], 'file':[], 'bow':[]}

			
			#songs_lyrics_ids_file.to_csv("integral_ClearSongsMoodsFile.csv")
		
		
		
		

		"""
		audios_not_found = list()
		for row_id, song in df.iterrows():
			video_id = song['youtube_video_id']
			ipdb.set_trace
			query_result = songs_lyrics_ids_file.query('youtube_video_id == @video_id')	
			if len(query_result) == 0:
				audios_not_found.append(song)
				#os.remove(audios_path+"/"+song['file'])
				#print("Video removed: ", song)
		"""



    