from datetime import datetime


class Logger:
    def __init__(self):
        self.date_time = datetime.today()  # obtained datetime object to extract current date and time
        self.date = self.date_time.date()
        self.cur_time = self.date_time.strftime('%H:%M:%S:%f')
        # write the relevant message in log file

    def log(self, file_object, message):
        """
            Method Name: log
            Description: This function write the progress or provided message of a particular operation
             in the _.txt file with its current date and time.
            Output: None
        """
        try:
            file_object.write(str(self.date) + "/" + str(self.cur_time) + "\t\t" + message + "\n")
        except OSError as s:
            file_object.write("Error during writing log: %s" %s)