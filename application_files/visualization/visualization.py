from datetime import datetime
import numpy as np
import pandas as pd
from cv2 import cv2
import os
from application_files.inference_graph.inference_graph import InferenceGraph
from application_files.model_file_operation.file_operation import FileOperationModelLoader
import matplotlib.pyplot as plt
import copy


class Visualization:
    def __init__(self, par_path, deviation, red_line_pos):
        self.par_path = par_path
        self.deviation = deviation
        self.red_line_pos = red_line_pos
        self.inference_graph = InferenceGraph()
        self.file_operation = FileOperationModelLoader()
        self.date_time = datetime.today()  # obtained datetime object to extract current date and time
        self.date = self.date_time.date()
        self.cur_time = self.date_time.strftime('%H_%M_%S_%f')

    def box_intersection_over_union(self, boxA, boxB, log, log_path):
        """
        Method Name: box_intersection_over_union
        Description: It calculates the relative portion of overlapping area between boxA and boxB with respect to
        the total area
        Output: area of overlap/area of union
        """
        log_file = open(log_path + 'visualization.txt', 'a+')  # open the log file
        try:
            intersection_width = np.maximum(0, (np.minimum(boxA[2], boxB[2]) - np.maximum(boxA[0], boxB[0])))
            intersection_height = np.maximum(0, (np.minimum(boxA[3], boxB[3]) - np.maximum(boxA[1], boxB[1])))
            intersection_area_AB = intersection_width * intersection_height  # Overlapping area of the boxes
            boxa_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])  # Total area of boxA
            boxb_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])  # Total area of boxB
            # write the log file
            log.log(log_file, 'Successfully calculated IOU using function box_intersection_over_union')
            log_file.close()
            return float(intersection_area_AB) / (boxa_area + boxb_area - intersection_area_AB)

        except Exception as e:
            # Write the log file for an error and close it
            log.log(log_file, 'Error during calculating IOU in function box_intersection_over_union ')
            log.log(log_file, str(e))
            log_file.close()

    def obj_box_draw(self, input_frame, obj, log, log_path, process0, model_licence, process1, model_licence_text, category_index_lic_text, flag_plate):
        """
            Method Name: obj_box_draw
            Description: This function draw the desired boxes on the predicted object and write few details about it.
            It also count the number of objects crossed the marked line and after crossing it saves a copy of
            that object.
            Output: count, input_frame
        """
        log_file = open(log_path + 'visualization.txt', 'a+')  # open the log file
        # initialize the color of rectangular box adn the text parameters
        box_color = (0, 255, 0)
        font_color = (0, 0, 0)
        font = 1
        font_size = 0.8
        # Save the coordinates of object box in local variables
        top = obj.box[0]
        left = obj.box[1]
        bottom = obj.box[2]
        right = obj.box[3]
        # Initialize the texts before writing them on the input frame
        text_1 = 'Obj. No.=' + str(obj.id)
        text_2 = 'Obj. Type=' + str(obj.class_name)
        text_3 = 'Percentage probability=' + str(round(100 * obj.score, 1)) + '%'
        # Draw box on input frame and write its details
        cv2.rectangle(input_frame, (left, top), (right, bottom), box_color, 2)
        cv2.putText(input_frame, text_1, (left, top - 35), font, font_size, font_color, 1, cv2.LINE_AA)
        cv2.putText(input_frame, text_2, (left, top - 25), font, font_size, font_color, 1, cv2.LINE_AA)
        cv2.putText(input_frame, text_3, (left, top - 15), font, font_size, font_color, 1, cv2.LINE_AA)

        #  Write the licence plate number if it is available
        if obj.lic_flag is not None and obj.lic_flag == True:
            cv2.putText(input_frame, 'Licence plate No.=' + str(obj.lic_plate), (left, top - 5), font, font_size,
                        (0, 0, 255), 1, cv2.LINE_AA)
        if obj.lic_flag is not None and obj.lic_flag == False:
            cv2.putText(input_frame, 'Licence plate No.=' + str(obj.lic_plate), (left, top - 5), font, font_size, font_color, 1, cv2.LINE_AA)
        #  Write the log file
        log.log(log_file, 'Successfully drawn the box and added text detail in visualization function')

        #  Compare the bottom coordinates to determine the object direction
        if bottom > obj.box[2]:
            direction = "down"
            count = 0
        else:
            direction = "up"
            count = 0

        #  Write the log file as a checkpoint
        log.log(log_file, 'Identified the object direction, in which it is going')

        try:
            # Check if object has crossed position of the marked line and save it's copy
            if abs(((bottom + top) / 2) - self.red_line_pos) < self.deviation:
                obj_capture = np.array(input_frame)
                # Extract the object based on the box coordinates
                obj_capture = obj_capture[int(top):int(bottom), int(left):int(right)]

                # Validate the object shape
                if obj_capture.shape[0] * obj_capture.shape[1] > 0 and obj_capture.shape[0] > 1 and obj_capture.shape[
                    1] > 1:
                    count = 1
                    filename = str(obj.class_name) + ' ID_No ' + str(obj.id) + ' is going in direction ' + \
                               str(direction) + ' on ' + str(self.date)+ ' at '+ str(self.cur_time) + '.png'
                    # Save the object as png file and write the log file
                    cv2.imwrite(os.path.join(self.par_path, 'vehicle_images', filename), obj_capture)
                    log.log(log_file, 'Object crossed the marked line and its image is saved as in .png format')

                    #  This section extract the information about the licence plate of the object
                    if process0 == 'Video+Licence plate':
                        lic_frame_expand = np.expand_dims(obj_capture, axis=0)
                        #  Predict the licence plate details from the frozen graph
                        lic_dic_out = self.inference_graph.run_inference_for_single_input_frame(model_licence,
                                                                                                lic_frame_expand,
                                                                                                log, log_path)
                        if len(lic_dic_out['detection_scores']) > 0:
                            max_index_score = np.argmax(lic_dic_out['detection_scores'])
                            lic_box = lic_dic_out['detection_boxes'][max_index_score]
                            lic_frame_x = [int(lic_box[0] * obj_capture.shape[0]),
                                           int(lic_box[2] * obj_capture.shape[1])]
                            lic_frame_y = [int(lic_box[1] * obj_capture.shape[0]),
                                           int(lic_box[3] * obj_capture.shape[1])]

                            # Only allows the right plate dimensions to be captured. if not correct than return the function
                            if lic_frame_x[0]<lic_frame_x[1] and lic_frame_y[0]<lic_frame_y[1]:
                                #  Extract the licence plate area from the overall object frame
                                lic_frame = obj_capture[min(lic_frame_x):max(lic_frame_x), min(lic_frame_y):max(lic_frame_y)]

                                lic_plate_filename = 'Licence plate for ' + str(obj.class_name) + ' ID_No ' + str(obj.id) \
                                                     + ' detected on' + str(self.date) + ' at ' + str(self.cur_time) + '.png'
                                # Save the licence plate image as in png file and write the log for a checkpoint
                                cv2.imwrite(os.path.join(self.par_path, 'vehicle_licence_plate_images', lic_plate_filename), lic_frame)
                                log.log(log_file, 'Number plate of vehicle identified and saved as in .png format')
                            else:
                                log.log(log_file, 'Predicted licence plate dimensions are not correct so next frame')
                                return count, input_frame

                        else:
                            log.log(log_file, 'Predicted dictionary for the licence plate is empty')

                        if process1 == 'Video+Licence plate+Text':
                            lic_text_frame_expand = np.expand_dims(lic_frame, axis=0)
                            #  Predict the details of the licence plate text from the frozen graph
                            output_dict_lic_text = self.inference_graph.run_inference_for_single_input_frame(
                                model_licence_text, lic_text_frame_expand,
                                log, log_path)
                            #  Initialize the local variables
                            classses_lic_text = []
                            box_lic_text = []
                            score_lic_text = []
                            height, width = lic_frame.shape[0], lic_frame.shape[1]

                            #  Extract the required key, value pair from the output dictionary
                            for key, value in enumerate(output_dict_lic_text['detection_scores']):
                                if value > 0.05:
                                    classses_lic_text.append(output_dict_lic_text['detection_classes'][key])
                                    box_lic_text.append(output_dict_lic_text['detection_boxes'][key])
                                    score_lic_text.append(value)
                            # Sort the boxes on the basis of left most coordinates and save sorted their index in a variable
                            idx_sort_lic_text = list(zip(*sorted([(val[1], i) for i, val in enumerate(box_lic_text)])))[
                                1]
                            lic_text_list = []
                            anchor_y = 0

                            #  Draw the boxes on the image characters and write the predicted value
                            for i, j in enumerate(idx_sort_lic_text):
                                if i == 0:
                                    if score_lic_text[j] > 0.1:
                                        text_top = int(box_lic_text[j][0] * height)
                                        text_left = int(box_lic_text[j][1] * width)
                                        text_bottom = int(box_lic_text[j][2] * height)
                                        text_right = int(box_lic_text[j][3] * width)
                                        anchor_y = abs(text_top + text_bottom) / 2
                                        lic_char = category_index_lic_text[classses_lic_text[j]]['name']
                                        cv2.rectangle(lic_frame, (text_left, text_top), (text_right, text_bottom),
                                                      box_color, 1)
                                        cv2.putText(lic_frame, str(lic_char), (text_left, text_bottom + 5), font,
                                                    font_size, font_color, 1,
                                                    cv2.LINE_AA)
                                        lic_text_list.append(lic_char)


                                else:
                                    s = self.box_intersection_over_union(box_lic_text[idx_sort_lic_text[i - 1]],
                                                                         box_lic_text[j], log, log_path)
                                    if s < 0.1 and score_lic_text[j] > 0.1:
                                        text_top = int(box_lic_text[j][0] * height)
                                        text_left = int(box_lic_text[j][1] * width)
                                        text_bottom = int(box_lic_text[j][2] * height)
                                        text_right = int(box_lic_text[j][3] * width)
                                        anchor2_y = abs(text_top + text_bottom) / 2
                                        if abs(anchor2_y - anchor_y) < 15:
                                            anchor_y = copy.copy(anchor2_y)
                                            lic_char = category_index_lic_text[classses_lic_text[j]]['name']
                                            cv2.rectangle(lic_frame, (text_left, text_top), (text_right, text_bottom),
                                                          box_color, 1)
                                            cv2.putText(lic_frame, str(lic_char), (text_left, text_bottom + 5), font,
                                                        font_size, font_color, 1, cv2.LINE_AA)
                                            lic_text_list.append(lic_char)
                            #  Save the licence plate image with predicted characters and then write the log file
                            cv2.imwrite(os.path.join(self.par_path, 'vehicle_licence_plate_text_images', lic_plate_filename), lic_frame)
                            log.log(log_file, 'Text has been predicted for the vehicle licence plate and saved in .png format')

                            lic_text_list = ''.join(str(i) for i in lic_text_list)
                            #  Initialize the dataframe that will contain the details of object if it is flagged or not
                            lic_text_results = pd.DataFrame(columns=['File_name', 'Date', 'Time', 'Licence_plate_no', 'Flagged'])
                            if flag_plate is not None and len(lic_text_list) > 0:
                                lic_text_csv_path = os.path.join(self.par_path, 'vehicle_licence_plate_csv/')
                                ratio = sum(map(lambda x, y: x == y, lic_text_list, flag_plate))/ len(flag_plate)

                                if ratio > 0.5:
                                    lic_results = lic_text_results.append({'File_name': lic_plate_filename, 'Date': str(self.date), 'Time': str(self.cur_time), 'Licence_plate_no': lic_text_list, 'Flagged': 'Yes'}, ignore_index=True)
                                    #  Write the results of predicted licence plate in the csv file
                                    lic_results.to_csv(lic_text_csv_path + 'licence_plate_result.csv', header=False, index=False, mode='a+')
                                    obj.lic_plate = lic_text_list
                                    obj.lic_flag = True
                                    cv2.putText(input_frame, 'Licence plate No.=' + str(lic_text_list),
                                                    (left, top - 5), font, font_size,
                                                    (0, 0, 255), 1, cv2.LINE_AA)
                                    log.log(log_file, 'Vehicle is flagged and not clear to pass')
                                    log.log(log_file, 'Predicted vehicle details are saved in the csv file')

                                else:
                                    lic_results = lic_text_results.append({'File_name': lic_plate_filename, 'Date': str(self.date), 'Time': str(self.cur_time), 'Licence_plate_no': lic_text_list, 'Flagged': 'No'}, ignore_index=True)
                                    #  Write the results of predicted licence plate in the csv file
                                    lic_results.to_csv(lic_text_csv_path + 'licence_plate_result.csv', header=False, index=False, mode='a+')
                                    obj.lic_plate = lic_text_list
                                    obj.lic_flag = False

                                    log.log(log_file, 'Vehicle is not flagged and it is clear to pass')
                                    log.log(log_file, 'Predicted vehicle details are saved in the csv file')

                            else:
                                print('Predicted Licence list is empty')
                        else:
                            print('Error during predicting the text for the licence plate')
                            log.log(log_file, 'Error during predicting the text for the licence plate')

                    else:
                        log.log(log_file, 'Error during the identification of the licence plate')

                else:
                    log.log(log_file, 'Object image is invalid')

            else:
                log.log(log_file, 'Nobody crossed the marked line')

            lic_text = [] # Initialize it to null so that the cycle can repeat
            log.log(log_file, 'Successfully exited the visualization')
            log_file.close()
            return count, input_frame

        except Exception as e:
            # Write the log file for an error and close it
            log.log(log_file, 'Error during processing visualization-obj_box_draw function')
            log.log(log_file, str(e))
            log_file.close()
