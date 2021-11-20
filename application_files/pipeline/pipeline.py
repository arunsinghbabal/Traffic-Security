import os
from application_files.log.logger import Logger
from application_files.create_directory.create_directory import DirectoryCreator
from application_files.traffic_tracking.traffic_tracking import TrafficTracking
from application_files.model_file_operation.file_operation import FileOperationModelLoader
from collections import deque
import collections


class Pipeline:
    def __init__(self, par_path, file_path):
        self.file_path = file_path
        self.par_path = par_path
        self.log = Logger()
        self.directory_creator = DirectoryCreator(self.par_path)
        self.file_operation = FileOperationModelLoader()

    def pipeline(self, operation=None, model_name=None, flag_plate=None):
        """
            Method Name: pipeline
            Description: This function validates and process the data and then train the models on it by calling
            the different functions
            Output: None
        """

        # Create necessary folders
        self.directory_creator.directory_creator(self.log)
        log_path = os.path.join(self.par_path, 'logger/')  # path to the logger folder
        log_file = open(log_path + 'pipeline.txt', 'a+')  # open the log file
        self.log.log(log_file, 'Directories created successfully')  # write the log file
        try:
            process0 = None
            process1 = None
            model_licence=None
            model_licence_text=None
            category_index_lic_text=None
            if model_name is None:
                model_name = 'VGG'
            if operation == 'Video+Licence plate+Text':
                process0 = 'Video+Licence plate'
                process1 = 'Video+Licence plate+Text'
                model_path = os.path.join(os.getcwd(), 'inference_graph', 'VGG', 'traffic_inference_graph')
                model, category_index = self.file_operation.load_model(model_path, self.log, log_path, True)
                model_lic_path = os.path.join(self.par_path,'inference_graph', model_name, 'licence_plate_inference_graph')
                model_licence = self.file_operation.load_model(model_lic_path, self.log, log_path, False)
                model_lic_text_path = os.path.join(self.par_path, 'inference_graph', 'VGG', 'licence_text_inference_graph')
                model_licence_text, category_index_lic_text = self.file_operation.load_model(model_lic_text_path, self.log, log_path, True)
                self.log.log(log_file, 'Model loaded successfully')

            elif operation == 'Video+Licence plate':
                model_path = os.path.join(os.getcwd(), 'inference_graph', 'VGG', 'traffic_inference_graph')
                model, category_index = self.file_operation.load_model(model_path, self.log, log_path, True)
                process0 = 'Video+Licence plate'
                model_lic_path = os.path.join(self.par_path, 'inference_graph', model_name, 'licence_plate_inference_graph')
                model_licence = self.file_operation.load_model(model_lic_path, self.log, log_path, False)
                self.log.log(log_file, 'Model loaded successfully')
            else:
                model_path = os.path.join(os.getcwd(), 'inference_graph', 'VGG', 'traffic_inference_graph')
                model, category_index = self.file_operation.load_model(model_path, self.log, log_path, True)
                self.log.log(log_file, 'Model loaded successfully')


            # Initialize the local variable
            obj_stk = []
            total_count = collections.defaultdict(int)
            id_storage = deque([i for i in range(100)])
            deviation = 5

            line_signal_counter = collections.defaultdict(int)
            red_line_pos = 437
            targeted_classes = ['car', 'truck']
            target_cls = [value['id'] for i, value in category_index.items() if
                          (targeted_classes[0] == value['name']) or (targeted_classes[1] == value['name'])]

            # Initialize the object for TrafficTracking class
            traffic_tracking = TrafficTracking(self.par_path, deviation, red_line_pos, category_index)

            # Track and count the traffic objects
            total_count = traffic_tracking.traffic_tracking(self.file_path, model, target_cls, line_signal_counter, red_line_pos,
                                              obj_stk, id_storage,total_count, self.log, log_path,
                                              process0, model_licence, process1, model_licence_text,
                                              category_index_lic_text, flag_plate)
            print('File analysis complete')
            self.log.log(log_file, 'Object tracking was successful')
            self.log.log(log_file, 'Pipeline execution was successful')
            log_file.close()
            return total_count

        except Exception as e:
            # Write the log file for an error and close it
            self.log.log(log_file, 'Pipeline execution was unsuccessful')
            self.log.log(log_file, str(e))
            log_file.close()
