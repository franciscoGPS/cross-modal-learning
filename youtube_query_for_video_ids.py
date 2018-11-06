import filesystem_tools as fs
import pandas as pd
from youtube_search import search_first_video_id, get_authenticated_service
import make_table as mt

class QueryVideo():

	def __init__(self):
		self.client = get_authenticated_service()

	def query_video(self, query_string):
		try:
			video_id =  search_first_video_id(self.client, part='snippet', maxResults=1, q=query_string, type='video')
			if video_id == '':
				video_id = 0
				print("Song not found: ", query_string)
		except IndexError as err:
			#ipdb.set_trace()
			video_id = 0
			print("Song not found: ", query_string)
		

		return video_id
		"""
		FILENAME=self.file
		OPENMODE=open_mode

		filesystem = fs.Filesystem(FILENAME, OPENMODE)

		songs = filesystem.read_csv_file_to_list("",FILENAME, OPENMODE)


		#for index, song in enumerate(len(songs['title'][:5])):
		#	print(song['title'][0], song['artist'][0])
		df = pd.DataFrame.from_dict(songs, orient='columns') 

		for row_id in df.index:
			query_string = df.loc[row_id]['title'] +" " +df.loc[row_id]['artist']
			try:
				video_id = search_first_video_id(client, part='snippet', maxResults=1, q=query_string, type='video')
				if (video_id != ""):
					df.loc[row_id,'youtube_id'] = video_id
				else:
					print("Song not found: ", query_string, row_id)	
			except IndexError as err:
				#ipdb.set_trace()
				print("Song not found: ", query_string, row_id)	



		df.to_csv('moods_songs_lyrics_with_youtube_ids_backup.csv')
		"""

