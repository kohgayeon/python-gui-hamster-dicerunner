import cv2
import numpy as np
import argparse
import time
import sys
from roboid import *
import math

class hamster_control:
    def __init__(self):
        self.marker_positions = [[] * 7 for i in range(7)]  # 2D List to store location information for all markers
        self.board_position = {}  # Dictionary to store the 6 marker positions on the board (fixed)
        self.hamster_position = {}  # Dictionary to store hamster location (continued update)
        self.dice_position = {}  # Dictionary to store dice location (continued update)

        self.dest = []  # Destination coordinate value for the hamster to arrive at
        self.distance = 0  # Distance between the hamster's current location and destination(dest)
        self.degree = 0  # for heading

        self.inter_stop = []  # 중간 정거장 (경로 중간에 주사위가 있는 경우)

    def cal_distance(self, route_mov):
        hpos = [self.hamster_position[2][0], self.hamster_position[2][1]]  # hamster's current location(hpos)
        if route_mov == 0:
            self.distance = math.dist(self.dest, hpos)  # To calculate the distance
        elif route_mov == 1:
            self.distance = math.dist(self.inter_stop, hpos)  # To calculate the distance

    def cal_heading(self, p1, p2):
        v1 = int((p1[0] + p2[0]) / 2.0)
        v2 = int((p1[1] + p2[1]) / 2.0)
        w = [v1, v2]  # heading_point: to find the direction of the hamster
        return w

    def rotation(self, route_mov):  # Function to find the direction the hamster will move
        distance1 = 0
        distance2 = 0
        if route_mov == 0:
            # If the hamster needs to rotate to the right, calculate the distance1
            head1 = Ham.cal_heading(self.hamster_position[2][5], self.hamster_position[2][7])
            distance1 = math.dist(self.dest, head1)

            # If the hamster needs to rotate to the left, calculate the distance2
            head2 = Ham.cal_heading(self.hamster_position[2][4], self.hamster_position[2][6])
            distance2 = math.dist(self.dest, head2)

        elif route_mov == 1:
            # If the hamster needs to rotate to the right, calculate the distance1
            head1 = Ham.cal_heading(self.hamster_position[2][5], self.hamster_position[2][7])
            distance1 = math.dist(self.inter_stop, head1)

            # If the hamster needs to rotate to the left, calculate the distance2
            head2 = Ham.cal_heading(self.hamster_position[2][4], self.hamster_position[2][6])
            distance2 = math.dist(self.inter_stop, head2)

        if distance1 < distance2:
            return 1  # Turn the hamster to the right
        elif distance2 < distance1:
            return 0  # Turn the hamster to the left

    def cal_angle(self, route_mov):  # Function to calculate the angle to find the right direction
        hpos = [self.hamster_position[2][0], self.hamster_position[2][1]]
        heading = self.hamster_position[2][3]

        if route_mov == 0:
            vector_1 = np.array([self.dest[0] - hpos[0], self.dest[1] - hpos[1]])
            vector_2 = np.array([heading[0] - hpos[0], heading[1] - hpos[1]])

            # Calculating the angle between the hamster and the destination
            innerAB = np.dot(vector_1, vector_2)
            AB = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)

            angle = np.arccos(innerAB / AB)
            self.degree = angle / np.pi * 180
        elif route_mov == 1:
            vector_1 = np.array([self.inter_stop[0] - hpos[0], self.inter_stop[1] - hpos[1]])
            vector_2 = np.array([heading[0] - hpos[0], heading[1] - hpos[1]])

            # Calculating the angle between the hamster and the destination
            innerAB = np.dot(vector_1, vector_2)
            AB = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)

            angle = np.arccos(innerAB / AB)
            self.degree = angle / np.pi * 180

    def cal_ab(self):   # 햄스터가 이동할 경로 (직선의 방정식) 계산하기
        x1, y1 = self.hamster_position[2][0], self.hamster_position[2][1]  # 햄스터 현재 좌표
        x2, y2 = self.dest[0], self.dest[1]                                # 목적지 좌표

        # y = ax + b
        if (x2 - x1) != 0:
            a = (y2 - y1)/(x2 - x1)
            b = (x2*y1 - x1*y2)/(x2 - x1)
            return a, b
        else:
            return x1, 0        # 햄스터와 목적지 x좌표가 동일함. (같은 x축 선상에 위치함.) x=x1

    def cal_inter(self, x, y):  # 중간 경로, (x, y): dice 좌표
        x1, y1 = self.hamster_position[2][0], self.hamster_position[2][1]  # 햄스터 현재 좌표
        x2, y2 = self.dest[0], self.dest[1]  # 목적지 좌표

        if abs(x1 - x) <= 40:               # 햄스터와 주사위가 x축선에 나란히 위치한 경우
            if x+150 < 800:                 # 전체 frame = [1000, 600]
                self.inter_stop = [x+150, y2]
            else:
                self.inter_stop = [x-150, y2]
        elif abs(y1 - y) <= 40:            # 햄스터와 주사위가 x축선에 나란히 위치한 경우
            if y+150 < 400:
                self.inter_stop = [x2, y+150]
            else:
                self.inter_stop = [x2, y-150]
        else:
            self.inter_stop = [x1, y2]     # 햄스터와 주사위가 x축과 y축에서 모두 나란히 있지 X

    def run(self):
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-t", "--type", type=str,
                        default="DICT_ARUCO_ORIGINAL",
                        help="type of ArUCo tag to detect")
        args = vars(ap.parse_args())

        # define names of each possible ArUco tag OpenCV supports
        ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }

        # verify that the supplied ArUCo tag exists and is supported by
        # OpenCV
        if ARUCO_DICT.get(args["type"], None) is None:
            print("[INFO] ArUCo tag of '{}' is not supported".format(
                args["type"]))
            sys.exit(0)

        # load the ArUCo dictionary and grab the ArUCo parameters
        print("[INFO] detecting '{}' tags...".format(args["type"]))
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
        arucoParams = cv2.aruco.DetectorParameters_create()
        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")

        vs = cv2.VideoCapture("http://192.168.66.1:9527/videostream.cgi?loginuse=admin&loginpas=admin")
        time.sleep(2.0)
        # Calculating the fps
        if vs.isOpened() == False:   # Checking the steady state of the camera
            print(f'Can\'t open the Camera')
            exit()

        prev_time = 0
        total_frames = 0
        start_time = time.time()

        initial_count = 0     # Variables to distinguish between scan stages
        dice_markerid = 7
        board_update = 0
        route_mov = 0         # 경로를 변경하기 위한 변수

        hamster = HamsterS()  # Connecting hamsterS

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 1000 pixels
            curr_time = time.time()
            retval, frame = vs.read()
            if not (retval):  # If frame information is not read normally
                break         # Escape the while

            total_frames = total_frames + 1

            term = curr_time - prev_time
            fps = 1 / term
            prev_time = curr_time
            fps_string = f'term = {term:.3f},  FPS = {fps:.2f}'

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                              # Change grayscale
            frame = cv2.resize(frame, dsize=(1000, 500), interpolation=cv2.INTER_CUBIC)  # Expand frame

            # Pincushion distortion
            # Setting the Distortion Factor ---①
            k1, k2, k3 = -0.065, 0, 0
            rows, cols = frame.shape[:2]

            # Creating a Mapping Array ---②
            mapy, mapx = np.indices((rows, cols), dtype=np.float32)

            # -1 to 1 normalization and polar coordinate conversion to center point coordinates ---③
            mapx = 2 * mapx / (cols - 1) - 1
            mapy = 2 * mapy / (rows - 1) - 1
            r, theta = cv2.cartToPolar(mapx, mapy)

            # Radial Distortion Transformation Operation ---④
            ru = r * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))

            # Restoring to orthogonal coordinates and top-left criteria ---⑤
            mapx, mapy = cv2.polarToCart(ru, theta)
            mapx = ((mapx + 1) * cols - 1) / 2
            mapy = ((mapy + 1) * rows - 1) / 2

            # Remapping ---⑥
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            # Perspective Transform
            pts1 = np.array([[159, 9], [159 + 684, 9], [159, 9 + 464], [159 + 684, 464]], np.float32)
            pts2 = np.array([[0, 0], [1000, 0], [0, 600], [1000, 600]], np.float32)

            M = cv2.getPerspectiveTransform(pts1, pts2)
            frame = cv2.warpPerspective(frame, M, (1000, 600))  # Size after conversion (x coordinate, y coordinate)

            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

            # Verify at least one ArUco marker was detected
            if len(corners) > 0:
                # Flatten the ArUco IDs list
                ids = ids.flatten()

                # Loop over the detected ArUCo corners
                # Multiple detection
                for (markerCorner, markerID) in zip(corners, ids):
                    if markerID != 0 and markerID != 1 and markerID != 2 and markerID != 3 and markerID != 4 and markerID != 5 and markerID != 6:
                        break                             # Since aruco markerid only uses 0, 1, 2, 3, 4, 5, 6

                    # Extract the marker corners
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    # Convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    # Calculate the diagonal of the marker
                    diagonal = int(math.dist(topLeft, bottomRight))
                    if diagonal < 60 and markerID != 2:    # If the sides of the dice are recognized
                        break

                    # Draw the bounding box of the ArUCo detection
                    cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

                    # Compute the center (x, y)-coordinates of the ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(frame, (cX, cY), 4, (0, 0, 0), -1)  # Black location coordinate point

                    # Draw the ArUco marker ID on the frame
                    cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)                # White markerID

                    if markerID == 2:
                        w = Ham.cal_heading(topRight, topLeft)  # Calculating the heading value of a hamster
                        # Save hamster marker position (opencv aruco marker id 2)
                        self.hamster_position[markerID] = (cX, cY, diagonal, w, topLeft, topRight, bottomLeft, bottomRight)

                    if markerID != 2 and initial_count != 0:         # Need to calculate the dice position after the the initial state
                        if not any(self.hamster_position):           # Exception handling if hamster robot may not be on board
                            print("Error 200: Hamster aruco marker not recognized")
                            continue
                        if markerID in self.board_position:
                            if (abs(self.board_position[markerID][0] - cX) >= 10 or abs(self.board_position[markerID][1] - cY) >= 10) and diagonal >= 90:  # When the position of the board aruco marker is changed during the play
                                self.board_position.update({markerID: (cX, cY, diagonal)})
                                board_update = 1

                        if dice_markerid == 7 and markerID in self.board_position:
                            if abs(self.board_position[markerID][0] - cX) >= 10 or abs(self.board_position[markerID][1] - cY) >= 10 or abs(self.board_position[markerID][2] - diagonal) >= 10:
                                if dice_markerid not in self.dice_position or board_update == 1:
                                    self.dice_position[markerID] = (cX, cY, diagonal)                                   # Dice is thrown -> update the dice position
                                    self.dest = [self.board_position[markerID][0], self.board_position[markerID][1]]    # Get the destination coordinate value for the hamster to arrive at
                                    dice_markerid = markerID

                                    # 햄스터가 장애물을 피해서 돌아가는 알고리즘 추가
                                    # 1) y=ax+b, 경로(직선 거리) 계산하기
                                    a, b = Ham.cal_ab()
                                    if b != 0:
                                        y = a * cX + b                # (cX, cY) = 주사위 위치 좌표

                                        if abs(y - cY) <= 52:         # 햄스터가 지나가야 할 경로에 주사위가 있음, 오차는 40(너비를 고려해서)
                                            route_mov = 1             # 경로를 수정함. (햄스터가 피해가야 함.)
                                            Ham.cal_inter(cX, cY)     # 지나가야 할 중간 정거장 계산하기
                                        else:
                                            if a != 0:
                                                x = (cY - b) / a
                                                if abs(x - cX) <= 52:  # 원래 45
                                                    route_mov = 1
                                                    Ham.cal_inter(cX, cY)  # 지나가야 할 중간 정거장 저장하기
                                                else:
                                                    route_mov = 0
                                            else:   # 기울기 a==0, y=b
                                                if abs(cY - b) <= 40:
                                                    route_mov = 1
                                                    Ham.cal_inter(cX, cY)  # 지나가야 할 중간 정거장 저장하기
                                                else:
                                                    route_mov = 0
                                    else:  # 햄스터와 목적지가 동일한 x축 선상에 위치함. (거의 희박한 경우) x=a
                                        if abs(cX - a) <= 40:
                                            route_mov = 1
                                            Ham.cal_inter(cX, cY)  # 지나가야 할 중간 정거장 저장하기
                                        else:
                                            route_mov = 0

                        # 1) Calculating the destination coordinate, hamster current position coordinate, distance value
                        if dice_markerid in self.dice_position:
                            Ham.cal_distance(route_mov)
                            cv2.circle(frame, self.dest, 4, (255, 255, 255), -1)       # Mark the coordinates of the destination with a white dot
                            if any(self.inter_stop):
                                cv2.circle(frame, self.inter_stop, 4, (255, 255, 255), -1)

                            # 2) Finding Hamster Orientation: Calculating the Angle
                            Ham.cal_angle(route_mov)
                            if self.distance < 20:   # Stop the hamster when it reaches its destination
                                hamster.stop()
                                dice_markerid = 7    # Initializing to roll the next dice
                                route_mov = 0
                                self.inter_stop.clear()
                                continue

                            if self.degree >= 15:
                                if Ham.rotation(route_mov) == 1:      # Turn right
                                    hamster.wheels(15, -15)
                                elif Ham.rotation(route_mov) == 0:    # Turn left
                                    hamster.wheels(-15, 15)

                                Ham.cal_angle(route_mov)
                                if self.degree < 15:
                                    hamster.wheels(30)     # Moving forward
                                    Ham.cal_distance(route_mov)
                                    if self.distance < 20:
                                        hamster.stop()
                                        dice_markerid = 7  # Initializing to roll the next dice
                                        route_mov = 0
                                        self.inter_stop.clear()
                            else:
                                # 3) Hamster Straight:
                                # Repeat until the distance to the destination approaches zero, continuing to detect the hamster's current position
                                hamster.wheels(30)         # Moving forward
                                Ham.cal_distance(route_mov)
                                if self.distance < 20:
                                    hamster.stop()
                                    dice_markerid = 7      # Initializing to roll the next dice
                                    route_mov = 0
                                    self.inter_stop.clear()
                    self.marker_positions[markerID].append((cX, cY, diagonal))  # Save location information for all recognized aruco markers

            initial_count = initial_count + 1
            cv2.putText(frame, fps_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # Initialize the location information of the six aruco markers (0, 1, 3, 4, 5, 6) on the board
            # Assume that there are no hamsters and dice on the board in the initial state
            for i in range(7):
                if i == 2:
                    self.board_position[i] = 0                  # Do not save to board list if hamster
                else:
                    if not any(self.marker_positions[i]):       # If there are no values in that list
                        print("Error 100: Six board aruco markers are not recognized")
                        # Initialization operation
                        initial_count = 0
                        for j in range(7):
                            self.marker_positions[j].clear()
                        self.board_position.clear()
                        break
                    for j in range(len(self.marker_positions[i])):
                        if self.marker_positions[i][j][2] >= 90 and board_update == 0:
                            self.board_position[i] = self.marker_positions[i][0]

                    if i not in self.board_position:  # 5번 board를 햄스터가 완전히 가리고, 주사위가 5번일 때 board에는 아무것도 저장되지 않음..
                        print("Error 100: Six board aruco markers are not recognized")
                        # Initialization operation
                        initial_count = 0
                        for j in range(7):
                            self.marker_positions[j].clear()
                        self.board_position.clear()
                        break

            # # Print the result
            # print("marker_positions:", self.marker_positions)
            # print("board_positions:", self.board_position)
            # print("dice_positions:", self.dice_position)

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        end_time = time.time()
        fps = total_frames / (start_time - end_time)
        print(f'total_frames = {total_frames},  avg FPS = {fps:.2f}')
        # do a bit of cleanup
        cv2.destroyAllWindows()

# Usage example
if __name__ == '__main__':
    Ham = hamster_control()
    Ham.run()