from application_files.pipeline.pipeline import Pipeline
import os
import sys

argument = sys.argv
file_path = os.path.join(os.getcwd(), 'input_video', 'input.mp4')
pipeline = Pipeline(os.getcwd(), file_path)
red_plate = 'HG53LLP'
if len(argument) == 4 and 'Video' in argument[1] and ('mobile' in argument[2] or 'VGG' in argument[2]):
    pipeline.pipeline(argument[1], argument[2], argument[3])
elif len(argument) == 3 and 'Video' in argument[1] and ('mobile' in argument[2] or 'VGG' in argument[2]):
    pipeline.pipeline(argument[1], argument[2])
elif len(argument) == 3 and 'Video' in argument[1] and ~('mobile' in argument[2] or 'VGG' in argument[2]):
    pipeline.pipeline(argument[1], None, argument[2])
else:
    pipeline.pipeline('Video')
