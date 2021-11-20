import os
from application_files.label_map_util.label_map_util import create_category_index_from_labelmap
import tensorflow as tf


class FileOperationModelLoader:
    def __init__(self):
        pass
    def load_model(self, model_path,log, log_path, category_index_value):
        """
            Method Name: load_model
            Description: This function loads the saved model
            Output: model
        """
        log_file = open(log_path + 'load_model.txt', 'a+')  # open the log file
        try:
            tf.gfile = tf.io.gfile
            model = tf.saved_model.load(model_path)  # load the custom model
            if category_index_value:
                label_path = os.path.join(model_path, 'label', 'labelmap.pbtxt')
                category_index = create_category_index_from_labelmap(label_path, use_display_name=True)
                log.log(log_file, 'Loaded model successfully')  # write the log
                log_file.close()
                return model, category_index
            else:
                log.log(log_file, 'Loaded model successfully')  # write the log
                log_file.close()
                return model

        except Exception as e:
            # Write error in log file and then close it
            log.log(log_file, ' model loading unsuccessful')
            log.log(log_file, str(e))
            log_file.close()

