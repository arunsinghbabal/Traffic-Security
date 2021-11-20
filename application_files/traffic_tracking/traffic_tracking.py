import os.path
from cv2 import cv2
import numpy as np
from application_files.inference_graph.inference_graph import InferenceGraph
from application_files.object_detection.object_detection import Detection
import time
import copy


class TrafficTracking:
    def __init__(self, par_path, deviation, red_line_pos, category_index):
        self.par_path = par_path
        self.inference_graph = InferenceGraph()
        self.detection = Detection(self.par_path, deviation, red_line_pos, category_index)

    def traffic_tracking(self, file_path, model, target_cls, line_signal_counter, red_line_pos, obj_stk, id_storage,
                         total_count, log, log_path, process0=None, model_licence=None, process1=None,model_licence_text=None,
                                              category_index_lic_text=None, flag_plate=None):
        """
            Method Name: traffic_tracking
            Description: This function process the input video file frame by frame and predict the objects using the
            inference model. It counts the number of objects crossing a specified line, write it on the frame and save
             those frames in the video format at the end.
            Output: None
        """
        log_file = open(log_path + 'traffictracking.txt', 'a+')  # open the log file

        # Open a video file
        cap = cv2.VideoCapture(file_path)
        #cap.set(cv2.CAP_PROP_FPS, 1)

        # Obtain its parameters
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps =int(cap.get(cv2.CAP_PROP_FPS))  # Frame per second
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Specify the video codec

        # Initialize the video writer with the parameters obtained from the input video
        output_video_path = os.path.join(self.par_path, 'output_video', 'the_output_new.avi')
        output_movie = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        try:
            while True:

                print('id_storage: {x}'.format(x=len(id_storage)))
                ret, frame = cap.read()  # read the frames from the input video
                # Break if no frame detected
                if not ret:
                    break

                input_frame = np.expand_dims(frame, axis=0)
                # Predict for the input frame
                output_dict = self.inference_graph.run_inference_for_single_input_frame(model, input_frame,
                                                                                        log, log_path)
                log.log(log_file, 'Object prediction in the input frame was successful')
                # Initialize the local variables
                scores = []
                boxes = []
                classes = []
                overall_total_count = 0

                # Extract only the targeted object classes and other parameters from the dictionary with a
                # minimum score of 0.6 to the local variables
                for t in target_cls:
                    for i, value in enumerate(output_dict['detection_classes']):
                        if (value == t) and (output_dict['detection_scores'][i] > 0.6):
                            scores.append(output_dict['detection_scores'][i])
                            boxes.append([int(output_dict['detection_boxes'][i][0] * height),
                                          int(output_dict['detection_boxes'][i][1] * width),
                                          int(output_dict['detection_boxes'][i][2] * height),
                                          int(output_dict['detection_boxes'][i][3] * width)])
                            classes.append(value)
                # Track the objects or items and update their status
                obj_stk, id_storage = self.detection.object_tracking(boxes, scores, classes, obj_stk, id_storage,
                                                                     log, log_path)
                log.log(log_file, 'Object tracking was successful')

                # Count the objects after they cross a specified line
                total_count, image, obj_stk, id_storage = self.detection.object_counting(frame, obj_stk, id_storage,
                                                                                         total_count, log, log_path,
                                                                                         process0, model_licence, process1,
                                                                                         model_licence_text,
                                                                                         category_index_lic_text,flag_plate)
                log.log(log_file, 'Object counted successfully')

                # Calculate the sum of all the objects that crossed the line
                for i, value in total_count.items():
                    overall_total_count = overall_total_count + int(value)

                # Write the overall_total_count value on the frame
                cv2.putText(image, 'Overall Count of Vehicles=' + str(overall_total_count), (5, 15), 5, 0.8,
                            (0, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

                # Write the individual class count on the frame
                for i in range(len(total_count)):
                    for m, n in total_count.items():
                        if m == 'car':
                            cv2.putText(image, 'Car' + '=' + str(n), (15, 35), 5, 0.8, (0, 0, 0xFF), 2,
                                        cv2.FONT_HERSHEY_SIMPLEX)
                        elif m == 'truck':
                            cv2.putText(image, 'Truck' + '=' + str(n), (15, 55), 5, 0.8, (0, 0, 0xFF), 2,
                                        cv2.FONT_HERSHEY_SIMPLEX)

                # Change the marked line color from green to blue, whenever object crosses it
                if overall_total_count == line_signal_counter[0]:
                    cv2.line(image, (0, red_line_pos), (width, red_line_pos), (0, 0xFF, 0), 5)
                else:
                    cv2.line(image, (0, red_line_pos), (width, red_line_pos), (0, 0, 0xFF), 5)
                    line_signal_counter[0] = line_signal_counter[0] + 1

                # Output frame in video format
                output_movie.write(image)
                log.log(log_file, 'Function traffic_tracking was successfully executed')
                log.log(log_file, '"writing frame..."')
                print("writing frame...")

            log.log(log_file, 'While loop in function traffic_tracking was successfully executed')
            log_file.close()
            return total_count

        except Exception as e:
            # Write the log file for an error and close it
            log.log(log_file, 'Error during traffic tracking')
            log.log(log_file, str(e))
            log_file.close()
