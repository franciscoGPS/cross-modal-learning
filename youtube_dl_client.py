from __future__ import unicode_literals
import youtube_dl
class YoutubeDLClient():
    """docstring for YoutubeDLClient"""
    def __init__(self, ydl_opts):
        ydl_opts.update({'logger': self.MyLogger()})
        ydl_opts.update({'progress_hooks': [self.my_hook]})
        self.ydl_opts = ydl_opts

        self.ydl = youtube_dl.YoutubeDL(ydl_opts)
        

    class MyLogger(object):
        def debug(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            print(msg)


    def my_hook(self, d):
        print(d['status'])
        if d['status'] == 'finished':
            print('Done downloading, now converting ...')


    def download_video(self, video_id):
        """
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            #'logger': MyLogger(),
            #'progress_hooks': [my_hook],
        }
        """
        #ydl_opts = {}
        #https://www.youtube.com/embed/chElHV99xak?start=53&end=59
        youtube_url = 'https://www.youtube.com/watch?v='
        self.ydl.download([youtube_url+video_id])
