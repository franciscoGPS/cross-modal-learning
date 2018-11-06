import dataset_verification as dataset
import subprocess
import os
 


AUDIO_STORAGE_PATH =  (os.path.abspath(os.path.dirname("__file__")) + '/songs/')
AUDIO_SYNCED_STORAGE_PATH =AUDIO_STORAGE_PATH + "synced/"
#subfolder_name = "30s_cuts/"
if not os.path.isdir(AUDIO_STORAGE_PATH):
		create_sub_dir = "mkdir "+ AUDIO_STORAGE_PATH
		subprocess.call(create_sub_dir, shell=True)



df = dataset.cleanup("integral_audios_dataset", AUDIO_STORAGE_PATH, AUDIO_SYNCED_STORAGE_PATH)
