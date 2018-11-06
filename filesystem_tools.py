import os.path as path
import csv
class Filesystem(object):
	#This method stores the passed data to the default video id's file

	def __init__(self, filename, open_mode):
		self.open_mode = open_mode
		self.file = filename
	#This method stores txt on the specified file with a specified EOL character. 
	def save_default( self, txt, EOL="" ):
		save_path = path.dirname(path.abspath(__file__))
		name_of_file = self.file
		completeName = path.join(save_path, name_of_file+".txt")
		file1 = open(completeName, self.open_mode)
		file1.write(txt+""+EOL+"\n")
		file1.close()

	def print_path(self):
		save_path = path.dirname(path.abspath(__file__))
		completeName = path.join(save_path, self.file+".txt")
		print(completeName)


	def file_to_list(self):
		#list_of_lists = []
		save_path = path.dirname(path.abspath(__file__))
		completeName = path.join(save_path, self.file+".txt")
		file1 = open(completeName, self.open_mode)
		#list_of_lists = file1.splitlines()
		list_of_songs = [line.split('\n') for line in file1.readlines()]
		file1.close()
		return list_of_songs

	def make_ids_table(self, file_name, array):
		save_path = path.dirname(path.abspath(__file__))
		completeName = path.join(save_path, self.file+".txt")
    	
		file1 = open(completeName, self.open_mode)
		csvWriter = csv.writer(file1,delimiter=',')
		csvWriter.writerows(array)
		file1.close()

	def get_songs_list(self):
		
		songs = list()
		with open(self.file, self.open_mode) as my_csv:
			song_list = csv.reader(my_csv)
			for row in song_list:
				songs.append(row)
		return songs	
    	



	def read_csv_file_to_list(self, file_path, file_name, open_mode):
		song_lyrics_list = {'mood':[], 'title':[], 'artist':[], 'lyric':[], 'youtube_id':[]}
		completeName = path.join(file_path, file_name+".csv")
		with open(completeName,open_mode) as my_csv:
			reader_list = csv.reader(my_csv)
			for row in reader_list:
				######
				"""
				songs structure
				[0] uselss id
				[1] artist
				[2] lyrics
				[3] mood
				[4] title
				"""
				######
				song_lyrics_list['mood'].append(str(row[3].strip()))
				song_lyrics_list['title'].append(str(row[4].strip()))
				song_lyrics_list['artist'].append(str(row[1].strip()))
				song_lyrics_list['lyric'].append(row[2].strip())
				song_lyrics_list['youtube_id'].append("")
		return song_lyrics_list	
    	