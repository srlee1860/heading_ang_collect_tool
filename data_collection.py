import os
import numpy as np
import cv2
import yaml
import copy

IMG_SIZE_WID = 700
IMG_SIZE_HEIGHT = 700

MIN_X = -35
MAX_X = 35
MIN_Y = -35
MAX_Y = 35
Z_FILTER = -1.5

DATASET_ROOT = "/Users/serin.lee/Works/mvt/semantic-kitti/dataset/sequences/01/"
LIDAR_FOLDER = DATASET_ROOT + "velodyne"
BOUNDING_BOX_LABEL = DATASET_ROOT + "labels"
CALIB_FOLDER = DATASET_ROOT + "calib"
DST_FOLDER = "./data"

tot = 7480
lidar_bb_png = None
lidar_bb_angle = None
cropped_vehicle = []
disp_cropped_vehicle = []
init_disp_cropped_vehicle = []
center_cv_x, center_cv_y = -1, -1
drawing = False

# open config file
try:
    CFG = yaml.safe_load(open("semantic-kitti.yaml", 'r'))
except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()
    
color_dict = CFG["color_map"]

max_sem_key = 0
for key, data in color_dict.items():
    if key + 1 > max_sem_key:
        max_sem_key = key + 1
sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
for key, value in color_dict.items():
    sem_color_lut[key] = np.array(value, np.float32) / 255.0

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def read_sem_label(label_filename):
    label = np.fromfile(label_filename, dtype=np.int32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF
    return sem_label

class kitti_object(object):
    '''Load and parse object data into a usable format.'''    
    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        #
        self.lidar_dir = LIDAR_FOLDER
        print (LIDAR_FOLDER, self.split_dir)
        self.label_dir = BOUNDING_BOX_LABEL
        self.calib_dir = os.path.join(CALIB_FOLDER, self.split_dir)

    def get_lidar(self, idx): 
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        print(lidar_filename)
        return load_velo_scan(lidar_filename), lidar_filename

    def get_sem_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, '%06d.label'%(idx))
        return read_sem_label(label_filename)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img

def click_and_crop(event, x, y, flags, param):
    global drawing, disp_cropped_vehicle, init_disp_cropped_vehicle

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.imshow('vehicle', init_disp_cropped_vehicle)
            disp_cropped_vehicle = copy.deepcopy(init_disp_cropped_vehicle)
            #print (center_cv_x, center_cv_y, x, y)
            cv2.line(disp_cropped_vehicle, (center_cv_x, center_cv_y), (x, 0), (255, 255, 255), 2)
            cv2.imshow('vehicle', disp_cropped_vehicle)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        heading_angle = np.arctan2(x - int(disp_cropped_vehicle.shape[1]/2), int(disp_cropped_vehicle.shape[0]/2))
        current_data = np.array([x, y, heading_angle])
        print('x = %d, y = %d, angle = %f' % (x, y, (180.0 / np.pi) * heading_angle))
        cv2.imwrite(os.path.join(DST_FOLDER, lidar_bb_png), cropped_vehicle)
        np.save(os.path.join(DST_FOLDER, lidar_bb_angle), current_data)

if __name__ == '__main__':
    cv2.namedWindow('vehicle')
    cv2.setMouseCallback('vehicle', click_and_crop)
    dataset = kitti_object(DATASET_ROOT, '')
    data_idx = 0
    nth_frame = 0
    filter_array = []

    while(True):
        lidar_data, lidar_data_name = dataset.get_lidar(data_idx)
        label_data = dataset.get_sem_label_objects(data_idx)
        print("lidar_data_name= ", lidar_data_name)
        print("lidar_data.shape= ", lidar_data.shape)
        print("label_data.shape= ", label_data.shape)
        
        sem_label_color = sem_color_lut[label_data]
        sem_label_color = sem_label_color.reshape((-1, 3))

        #BEV
        inImg_ = np.zeros((IMG_SIZE_WID, IMG_SIZE_HEIGHT, 3), np.uint8)
        
        idx_ = np.where( (lidar_data[:, 0] > MIN_X) & (lidar_data[:, 0] < MAX_X) 
                        & (lidar_data[:, 1] < MAX_Y) & (lidar_data[:, 1] < MAX_Y)
                        & (lidar_data[:, 2] > Z_FILTER) 
                        )
        i_ = (lidar_data[:, 0][idx_]-(MIN_X)) * 10
        j_ = (lidar_data[:, 1][idx_] -(MIN_Y)) * 10

        red_ = sem_label_color[:, 0][idx_] * 255
        blue_ = sem_label_color[:, 1][idx_]* 255
        green_ = sem_label_color[:, 2][idx_]* 255

        inImg_[j_.astype(int), i_.astype(int), 0] = blue_.astype(int)
        inImg_[j_.astype(int), i_.astype(int), 1] = green_.astype(int)
        inImg_[j_.astype(int), i_.astype(int), 2] = red_.astype(int)

        # collect the point cloud included in "car"
        # need to FIX
        car_pt = np.zeros_like(inImg_)
        car_pt[inImg_[:,:,2] == 255] = [255, 255, 255]

        # get more uniform "car" point cloud, the dilation operator is applied. may need to be tuned
        # result_gray represents the processed "car" point cloud
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.dilate(car_pt, kernel, iterations=2)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result_gray = cv2.threshold(result_gray, 127, 255, cv2.THRESH_BINARY)[1]

        # get connected component from the car point cloud 
        num_labels, labels_im = cv2.connectedComponents(result_gray)
        labeled_img = imshow_components(labels_im)
        gray = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)

        # get contours
        contours, hier = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        bbox_idx = 0
        bbox_max = len(contours) - 1
        while(True):
            if bbox_idx > bbox_max:
                break
            print("current box_id, # of boxes = ", bbox_idx, bbox_max)

            (x, y, w, h) = cv2.boundingRect(contours[bbox_idx])

            # filter out small height box
            if h < 20:
                bbox_idx += 1
                if bbox_idx == bbox_max:
                    break
                else:
                    continue

            # crop vehicles
            cropped_vehicle = car_pt[y:y+h, x:x+w, :]
            disp_cropped_vehicle = cv2.resize(copy.deepcopy(cropped_vehicle), (0,0), fx=2.0, fy=2.0)
            init_disp_cropped_vehicle = copy.deepcopy(disp_cropped_vehicle)

            # draw bounding box
            cv2.rectangle(inImg_, (x, y), (x + w, y + h), (255, 255, 255), 1)

            # draw line in cropped vehicle
            center_cv_x = int(disp_cropped_vehicle.shape[1] / 2)
            center_cv_y = int(disp_cropped_vehicle.shape[0] / 2)
            cv2.line(disp_cropped_vehicle, (center_cv_x, center_cv_y), (center_cv_x, 0), (255, 255, 255))

            # display windows
            cv2.imshow('vehicle', disp_cropped_vehicle)
            cv2.imshow("bev", inImg_)

            lidar_bb_png = os.path.basename(lidar_data_name).split(".")[0] + "_" + str(bbox_idx) + ".png"
            lidar_bb_angle = os.path.basename(lidar_data_name).split(".")[0] + "_" + str(bbox_idx) + ".npy"

            key_b = cv2.waitKey(0)
            if key_b == ord('a') or key_b == ord('A'): # next bounding box
                bbox_idx += 1
            elif key_b == ord('b') or key_b == ord('B'): # previous bounding box
                bbox_idx -= 1
                if bbox_idx < 0:
                    bbox_idx = 0
            if key_b == ord('c') or key_b == ord('C'):  # exit
                break

        nth_frame += 1
        print ("***********************************************\n")
        print (" Next : n \n Back : b \n Quit : q \n Jump : j")
        key = cv2.waitKey(0) #change to your own waiting time 1000 = 1 second 
        print (key)
        if key == 27 or key == ord('q') or key == ord('Q'): #if ESC is pressed, exit
            print("Quit")
            cv2.destroyAllWindows()
            break
        elif key == ord('n') or key == ord('N') or key == ord('z') or key == ord('Z'): # next
            print("Next")
            data_idx = data_idx + 1
            if data_idx > tot:
                data_idx = tot
        elif key == ord('b') or key == ord('B'): # previous
            print("Previous")
            data_idx = data_idx - 1
            if data_idx <= 0:
                data_idx = 0
        elif key == ord('j') or key == ord('J') or key == ord('i') or key == ord('I'): # jump
            print("Jump Index. \n Enter without zeros")
            mode_index = int(input(" Enter index ? : "))
            if mode_index >= 0 and mode_index < tot:
                data_idx = mode_index
