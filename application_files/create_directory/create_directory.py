import os


class DirectoryCreator:
    def __init__(self, par_path):
        self.par_path = par_path

    def directory_creator(self, log):
        """
            Method Name: directory_creator
            Description: This function creates all the necessary folders essential for the project operation,
            if they are non-existent.
            Output: None
        """
        # assign the possible paths to the related variables
        output_video = os.path.join(self.par_path, 'output_video')
        vehicle_images = os.path.join(self.par_path, 'vehicle_images')
        vehicle_licence_plate_images = os.path.join(self.par_path, 'vehicle_licence_plate_images')
        vehicle_licence_plate_text_images = os.path.join(self.par_path, 'vehicle_licence_plate_text_images')
        vehicle_licence_plate_csv = os.path.join(self.par_path, 'vehicle_licence_plate_csv')
        logger = os.path.join(self.par_path, 'logger')

        # First create a folder to log all the details
        if not os.path.isdir(logger):
            os.makedirs(logger)
        # Open the log files
        log_file = open(logger + '/folder_preparation.txt', 'a+')

        try:
            # Verify if the folder exist or not. If it does not exist than create it.
            if not os.path.isdir(output_video):
                os.makedirs(output_video)
            if not os.path.isdir(vehicle_images):
                os.makedirs(vehicle_images)
            if not os.path.isdir(vehicle_licence_plate_images):
                os.makedirs(vehicle_licence_plate_images)
            if not os.path.isdir(vehicle_licence_plate_text_images):
                os.makedirs(vehicle_licence_plate_text_images)
            if not os.path.isdir(vehicle_licence_plate_csv):
                os.makedirs(vehicle_licence_plate_csv)

            # Call the log function to append the success message
            log.log(log_file, 'Folder created successfully')
            log_file.close()  # close the log file

        except OSError as s:
            # Call the log function to append the error message
            log.log(log_file, 'Error during folder creation: %s' % s)
            log_file.close()  # close the log file




