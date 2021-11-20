import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag


class Tracker():  # class for Kalman Filter based tracker
    def __init__(self, x_state, id, score, classes):
        # Initialize parameters for tracker (history)
        self.id = id  # tracker's id
        self.box = []  # list to store the coordinates for a bounding box
        self.hits = 0  # number of detection matches
        self.no_losses = 0  # number of unmatched tracks (track loss)
        self.score = score
        self.class_name = classes
        self.lic_plate = None
        self.lic_flag = None

        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        self.x_state = x_state
        self.bottom_pos = 0
        self.dt = 1.  # time interval

        # Process matrix, assuming constant velocity model
        self.F = np.array([[1, 0, 0, 0, self.dt, 0, 0, 0],
                           [0, 1, 0, 0, 0, self.dt, 0, 0],
                           [0, 0, 1, 0, 0, 0, self.dt, 0],
                           [0, 0, 0, 1, 0, 0, 0, self.dt],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

        # Measurement matrix, assuming we can only measure the coordinates
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0]])

        # Initialize the state covariance
        self.L = 100.0
        self.P = np.diag(self.L * np.ones(8))

        # Initialize the process covariance
        self.Q_comp_mat = np.array([[self.dt ** 4 / 2., self.dt ** 3 / 2.], [self.dt ** 3 / 2., self.dt ** 2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat, self.Q_comp_mat, self.Q_comp_mat)

        # Initialize the measurement covariance
        self.R_ratio = 1.0 / 16.0
        self.R_diag_array = self.R_ratio * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)


    def kalman_filter(self, z, log, log_path):
        """
            Method Name: kalman_filter
            Description: This function calculate the kalman gain and error covariance using the measured and estimated
            box coordinate values. It gives the new box coordinates for the object.
            Output: box
        """
        log_file = open(log_path + 'tracker.txt', 'a+')  # open the log file
        try:
            x = self.x_state
            # Predict
            x = dot(self.F, x)
            self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

            # Update
            S = dot(self.H, self.P).dot(self.H.T) + self.R
            K = dot(self.P, self.H.T).dot(inv(S))  # Kalman gain
            y = np.expand_dims(z, axis=0).T - dot(self.H, x)  # residual
            x += dot(K, y)
            self.P = self.P - dot(K, self.H).dot(self.P)
            self.x_state = x.astype(int)
            # New predicted box coordinates
            self.box = [self.x_state[0][0], self.x_state[1][0], self.x_state[2][0], self.x_state[3][0]]
            log.log(log_file, 'Successfully predicted the new box coordinates from the kalman_filter function')
            log_file.close()
            return self.box

        except Exception as e:
            # Write the log file for an error and close it
            log.log(log_file, 'Error during the prediction of new box coordinates in trackers function kalman_filter ')
            log.log(log_file, str(e))
            log_file.close()

    def predict_only(self,log,log_path):
        """
            Method Name: predict_only
            Description: This function only predict the box coordinates based on the kalman gain and covariance from
            the previous measurements
            Output: box
        """
        log_file = open(log_path + 'tracker.txt', 'a+')  # open the log file

        try:
            x = self.x_state
            # Predict
            x = dot(self.F, x)
            # covariance
            self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
            self.x_state = x.astype(int)
            # Estimate box coordinates
            self.box = [self.x_state[0][0], self.x_state[1][0], self.x_state[2][0], self.x_state[3][0]]
            log.log(log_file, 'Successfully predicted the new box coordinates from the predict_only function')
            log_file.close()
            return self.box

        except Exception as e:
            # Write the log file for an error and close it
            log.log(log_file, 'Error during the prediction of box coordinates in trackers function predict_only')
            log.log(log_file, str(e))
            log_file.close()
