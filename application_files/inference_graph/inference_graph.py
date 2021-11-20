import numpy as np
import tensorflow as tf


class InferenceGraph:
    def __init__(self):
        pass

    def run_inference_for_single_input_frame(self, model, input_frame,log, log_path):
        """
            Method Name: run_inference_for_single_input_frame
            Description: This function make prediction on the given input frame and provides us the results 
            in a dictionary format
            Output: output_dict
        """

        log_file = open(log_path + 'run_inference_for_single_input_frame.txt', 'a+')
        try:
            input_tensor = tf.convert_to_tensor(input_frame)
            # Initialize the model with a default set of data attributes that were used to build it
            model_fn = model.signatures['serving_default']
            # Make predictions for the input_frame from the model
            output_dict = model_fn(input_tensor)
            # Took out the num_detection from dictionary because od its 1D shape=(1,)
            num_detections = int(output_dict.pop('num_detections'))

            # Convert the output dictionary tensor values in numpy array
            output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
            output_dict['num_detections'] = num_detections
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int16)

            log.log(log_file, 'Prediction from the input frame was successful')
            log_file.close()

            return output_dict

        except Exception as e:
            log.log(log_file, 'Error during prediction from the input frame')
            log.log(log_file, str(e))
            log_file.close()

