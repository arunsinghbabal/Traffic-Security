import numpy as np
from application_files.tracker.tracker import Tracker
from application_files.visualization.visualization import Visualization
from scipy.optimize import linear_sum_assignment


class Detection:
    def __init__(self, par_path, deviation, red_line_pos, category_index):
        self.par_path = par_path
        self.category_index = category_index
        self.visualization = Visualization(self.par_path, deviation, red_line_pos)

    def object_tracking(self, boxes, scores, classes, obj_stk, id_storage, log, log_path):
        """
            Method Name: obj_box_draw
            Description: This function matches the  boxes with the previous ones using the Tracker class and identifies
            the matched, vanished and newly detected boxes. It also updates the object list for the continuation
            of the cycle
            Output: obj_stk, id_storage
        """
        log_file = open(log_path + 'detection.txt', 'a+')  # open the log file

        # initialize the local variables
        new_boxes = boxes
        old_boxes = []
        matches = []
        vanished = []
        new_detected = []

        try:

            # Fill the old box coordinates using the object stock list
            if len(obj_stk) > 0:
                for i in obj_stk:
                    old_boxes.append(i.box)

            iou_mat = np.zeros((len(old_boxes), len(new_boxes)), dtype=np.float16)

            # Create box intersection over union matrix for the old and new boxes
            for i in range(len(old_boxes)):
                for j in range(len(new_boxes)):
                    iou_mat[i, j] = self.visualization.box_intersection_over_union(old_boxes[i], new_boxes[j],
                                                                                   log, log_path)

            # Find the updated ID of the old boxes in the new frame by assigning the ID in a way that it maximizing
            # the total area or job
            old_id, new_id = linear_sum_assignment(iou_mat, maximize=True)
            old_new_id = np.stack((old_id, new_id), axis=1)

            # Update the local variables for the vanished, newly detected and matched boxes ID
            for i in range(len(old_boxes)):
                if i not in old_new_id[:, 0]:
                    vanished.append(i)
            for i in range(len(new_boxes)):
                if i not in old_new_id[:, 1]:
                    new_detected.append(i)
            for i in old_new_id:
                if iou_mat[i[0], i[1]] > 0.3:
                    matches.append(i)
                else:
                    vanished.append(i[0])
                    new_detected.append(i[1])

            log.log(log_file, 'Updated the local variables for matched, vanished and newly detected objects')

            # Create tracker object for newly detected items and add the obtained box coordinates in old_boxes variable
            if len(new_detected) > 0:
                for i in new_detected:
                    x_state = np.hstack(([new_boxes[i]], [[0, 0, 0, 0]])).T
                    # Initialize the tracking class object, which is unique to the newly detected item
                    tracker = Tracker(x_state, id_storage.popleft(), scores[i], self.category_index[classes[i]]['name'])
                    # Initialize the items initial state by providing box coordinates and estimate error covariance
                    # matrix, which will be used for kalman gain
                    box = tracker.predict_only(log, log_path)
                    # append the box coordinates, which was obtained by applying a dot product between the previous box
                    # coordinate and constant velocity. It allows us to estimate a new coordinates for the item in
                    # next frame.
                    old_boxes.append(box)
                    obj_stk.append(tracker)
                log.log(log_file, 'Object stock updated for newly detected items ')
            # Time dependent updating of error covariance for the vanished item in tracking class and identify the
            # new box coordinates while adding one in its no_losses variable
            if len(vanished) > 0:
                for i in vanished:
                    van_obj = obj_stk[i]  # obtain the tracking class object related to the vanished item
                    van_obj.no_losses += 1  # Shows the number of times the item is not appeared in the model prediction
                    old_boxes[i] = van_obj.predict_only(log, log_path)  # update the estimated new box coordinates
                log.log(log_file, 'Object stock updated for vanished items ')
            # Update the error covariance based on the estimated and measured coordinates of the item using the kalman
            # gain and add 1 to show how many times the item has detected successfully
            if len(matches) > 0:
                for i, j in matches:
                    match_obj = obj_stk[i]  # Obtain the item object
                    box = match_obj.kalman_filter(new_boxes[j], log, log_path)
                    match_obj.hits += 1  # showing the nos of times item detected
                    old_boxes[i] = box
                log.log(log_file, 'Object stock updated for matched items ')#
            log.log(log_file, 'Function object_tracking executed successfully')
            log_file.close()
            return obj_stk, id_storage

        except Exception as e:
            # Write the log file for an error and close it
            log.log(log_file, 'Error during the operation of function object_tracking ')
            log.log(log_file, str(e))
            log_file.close()

    def object_counting(self, input_frame, obj_stk, id_storage, total_count,log, log_path, process0, model_licence, process1,model_licence_text, category_index_lic_text, flag_plate):
        """
            Method Name: object_counting
            Description: This function removes the item object from the stock if it does' show up to 10 times and add
            its ID to the id storage. It also updates the every item's category count by adding 1 each time the object
            crosses the marked line
            Output: total_count, input_frame,obj_stk, id_storage
        """

        log_file = open(log_path + 'detection.txt', 'a+')  # open the log file

        try:
            for obj in obj_stk:
                if (obj.hits >= 1) and (obj.no_losses <= 10):
                    count, input_frame = self.visualization.obj_box_draw(input_frame, obj, log, log_path, process0, model_licence, process1, model_licence_text,category_index_lic_text, flag_plate)
                    total_count[obj.class_name] = total_count[obj.class_name] + count
                    log.log(log_file, 'Updated the total count for object class')

                if obj.no_losses > 10:
                    id_storage.append(obj.id)
                    obj_stk.remove(obj)
                    log.log(log_file, 'Object removed from the stock')

            log.log(log_file, 'Successfully counted the objects')
            log_file.close()
            return total_count, input_frame,obj_stk, id_storage

        except Exception as e:
            # Write the log file for an error and close it
            log.log(log_file, 'Error during the operation of function object_counting ')
            log.log(log_file, str(e))
            log_file.close()

