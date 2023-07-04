import cv2
import numpy as np
import argparse
import time
import sys
from roboid import *
import math

marker_positions = [[] * 7 for i in range(7)]  # 모든 마커의 위치 정보를 저장할 2차원 list
board_position = {}                            # 보드판에 있는 6개의 마커 위치를 저장할 딕셔너리 (고정)
hamster_position = {}                          # 햄스터의 위치 정보를 저장할 딕셔너리 (계속 업데이트)
dice_position = {}                             # 주사위의 위치 정보를 저장할 딕셔너리 (계속 업데이트)

class hamster_control:
    def __init__(self):
        self.dest = []     # 햄스터가 도착해야 할 목적지 좌표값
        self.distance = 0  # 거리
        self.degree = 0    # 각도

    def cal_distance(self):
        hpos = [hamster_position[2][0], hamster_position[2][1]]  # 햄스터 현재 위치 좌표값(hpos) 가져오기
        self.distance = math.dist(self.dest, hpos)  # (목적지-햄스터) 거리 계산 함수

    def cal_heading(self, p1, p2):
        v1 = int((p1[0] + p2[0]) / 2.0)
        v2 = int((p1[1] + p2[1]) / 2.0)
        w = [v1, v2]  # heading_point: 햄스터의 방향을 찾기 위한 포인트
        return w

    def rotation(self):             # 햄스터 방향 찾는 동작 함수
        # 경우1) topLeft가 topRight 위치로 간 경우(오른쪽 회전) -> heading_point가 topRight과 bottomRight의 중간이 된다.
        head1 = Ham.cal_heading(hamster_position[2][5], hamster_position[2][7])
        distance1 = math.dist(self.dest, head1)  # 햄스터의 heading_point와 목적지 간의 거리 계산

        # 경우2) topRight가 topLeft 위치로 간 경우(왼쪽 회전) -> heading_point가 topLeft와 bottomLeft의 중간이 된다.
        head2 = Ham.cal_heading(hamster_position[2][4], hamster_position[2][6])
        distance2 = math.dist(self.dest, head2)

        if distance1 < distance2:    # 경우1: 오른쪽으로 회전해야 함
            return 1
        elif distance2 < distance1:  # 경우2: 왼쪽으로 회전해야 함
            return 0

    def cal_angle(self):  # 각도 계산 함수(목적지, 햄스터 중앙점, 햄스터 heading)
        hpos = [hamster_position[2][0], hamster_position[2][1]]
        heading = hamster_position[2][3]

        vector_1 = np.array([self.dest[0] - hpos[0], self.dest[1] - hpos[1]])
        vector_2 = np.array([heading[0] - hpos[0], heading[1] - hpos[1]])

        # Angle
        innerAB = np.dot(vector_1, vector_2)
        AB = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
        angle = np.arccos(innerAB / AB)
        self.degree = angle / np.pi * 180

    def run(self):
        vs = cv2.VideoCapture("http://192.168.66.1:9527/videostream.cgi?loginuse=admin&loginpas=admin")
        time.sleep(2.0)
        # fps 계산
        if vs.isOpened() == False:  # 카메라 정상상태 확인
            print(f'Can\'t open the Camera')
            exit()

        prev_time = 0
        total_frames = 0
        start_time = time.time()
        initial_count = 0
        dice_markerid = 7

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 1000 pixels
            curr_time = time.time()
            retval, frame = vs.read()
            if not (retval):  # 프레임정보를 정상적으로 읽지 못하면
                break         # while문을 빠져나가기

            total_frames = total_frames + 1

            term = curr_time - prev_time
            fps = 1 / term
            prev_time = curr_time
            fps_string = f'term = {term:.3f},  FPS = {fps:.2f}'
            # print(fps_string)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
            frame = cv2.resize(frame, dsize=(1000, 500), interpolation=cv2.INTER_CUBIC)  # 이미지 확대

            # 핀큐션 왜곡
            # 왜곡 계수 설정 ---①
            k1, k2, k3 = -0.065, 0, 0
            rows, cols = frame.shape[:2]

            # 매핑 배열 생성 ---②
            mapy, mapx = np.indices((rows, cols), dtype=np.float32)

            # 중앙점 좌표로 -1~1 정규화 및 극좌표 변환 ---③
            mapx = 2 * mapx / (cols - 1) - 1
            mapy = 2 * mapy / (rows - 1) - 1
            r, theta = cv2.cartToPolar(mapx, mapy)

            # 방사 왜곡 변영 연산 ---④
            ru = r * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))

            # 직교좌표 및 좌상단 기준으로 복원 ---⑤
            mapx, mapy = cv2.polarToCart(ru, theta)
            mapx = ((mapx + 1) * cols - 1) / 2
            mapy = ((mapy + 1) * rows - 1) / 2
            # 리매핑 ---⑥
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            # perspective transform
            pts1 = np.array([[169, 0], [169 + 701, 0], [169, 0 + 490], [169 + 701, 490]], np.float32)
            pts2 = np.array([[0, 0], [1000, 0], [0, 600], [1000, 600]], np.float32)  # 왼쪽위점, 오른쪽위점, 왼쪽아래점, 오른쪽아래점

            M = cv2.getPerspectiveTransform(pts1, pts2)
            frame = cv2.warpPerspective(frame, M, (1000, 600))  # 변환후 크기 (x좌표, y좌표)

            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

            # Verify at least one ArUco marker was detected
            if len(corners) > 0:
                # Flatten the ArUco IDs list
                ids = ids.flatten()
                if initial_count == 0 and len(ids) != len(set(ids)):  # 초기 상태일 때 중복된 markerID가 잡혔는지 확인
                    print("Error 300: 초기에 주사위 aruco marker가 인식된 경우 다시 인식하기")
                    # 초기화 작업
                    initial_count = 0
                    for j in range(7):
                        marker_positions[j].clear()
                    board_position.clear()
                    continue
                # Loop over the detected ArUCo corners
                # 다중 detection!! -> 한 번에 여러 개의 마커를 잡음
                for (markerCorner, markerID) in zip(corners, ids):
                    if markerID != 0 and markerID != 1 and markerID != 2 and markerID != 3 and markerID != 4 and markerID != 5 and markerID != 6:
                        break      # aruco markerid가 0, 1, 2, 3, 4, 5, 6만 사용하기 때문에

                    # Extract the marker corners
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    # Convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    # Calculate the height of the marker
                    height = abs(int(topLeft[1]) - int(bottomRight[1]))

                    # Draw the bounding box of the ArUCo detection
                    cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

                    # Compute the center (x, y)-coordinates of the ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

                    # Draw the ArUco marker ID on the frame
                    cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                    if markerID == 2:   # 햄스터의 heading 값 계산하기: 햄스터의 방향 찾기
                        w = Ham.cal_heading(topRight, topLeft)
                        # 햄스터 마커 위치 저장(opencv aruco marker id 2)
                        hamster_position[markerID] = (cX, cY, height, w, topLeft, topRight, bottomLeft, bottomRight)

                    # 주사위 위치 update
                    if markerID != 2 and initial_count != 0:    # 초기화 단계가 지난 후에 주사위 위치를 계산해야 함
                        if not any(hamster_position):           # 햄스터 로봇이 board 위에 없을 수 있는 경우 예외처리
                            print("Error 200: 햄스터 aruco marker가 인식되지 않는 경우")
                            continue

                        if dice_markerid == 7:
                            if abs(board_position[markerID][0] - cX) >= 10 or abs(board_position[markerID][1] - cY) >= 10 or abs(board_position[markerID][2] - height) >= 10:
                                if dice_markerid not in dice_position:
                                    dice_position[markerID] = (cX, cY, height)  # 주사위가 던져졌음
                                    for key, value in board_position.items():
                                        if markerID == key:
                                            self.dest = [value[0], value[1]]    # 햄스터가 도착해야할 목적지 좌표값(dest = cX, cY) 가져오기
                                    dice_markerid = markerID

                        if dice_markerid in dice_position:
                            Ham.cal_distance()      # 1) 목적지 좌표값, 햄스터 현위치 좌표값, 그 거리값 계산하고
                            cv2.circle(frame, self.dest, 4, (255, 0, 0), -1)  # 목적지의 좌표값 표시함

                            # 2) 햄스터 방향 찾기: 각도 계산하기
                            Ham.cal_angle()
                            if self.distance < 20:  # 햄스터가 목적지에 도달하면 멈추기
                                hamster.stop()
                                dice_markerid = 7   # 다음 주사위를 위해 초기화하기
                                continue

                            if self.degree >= 15:
                                if Ham.rotation() == 1:      # turn right
                                    hamster.wheels(15, -15)  # (왼쪽 바퀴, 오른쪽 바퀴)
                                elif Ham.rotation() == 0:    # turn left
                                    hamster.wheels(-15, 15)

                                Ham.cal_angle()
                                if self.degree < 15:
                                    hamster.wheels(30)  # 앞으로 이동하기
                                    Ham.cal_distance()
                                    if self.distance < 20:
                                        hamster.stop()
                                        dice_markerid = 7  # 다음 주사위를 위해 초기화하기
                            else:
                                # 3) 햄스터 직진: 계속 햄스터의 현위치를 감지하면서 목적지와의 거리를 계산해야 함, 거리가 0에 수렴할 때까지 반복
                                hamster.wheels(30)  # 앞으로 이동하기
                                Ham.cal_distance()
                                if self.distance < 20:
                                    hamster.stop()
                                    dice_markerid = 7  # 다음 주사위를 위해 초기화하기
                    marker_positions[markerID].append((cX, cY, height))  # 2차원 리스트 marker_positions에 튜플 형태로 위치정보 저장하기

            initial_count = initial_count + 1
            cv2.putText(frame, fps_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # 보드판에 있는 6개의 aruco marker(0, 1, 3, 4, 5, 6)의 위치 정보 초기화
            for i in range(7):
                if i == 2:                                      # 햄스터의 aruco marker인 경우
                    board_position[i] = 0                       # board aruco marker 저장 X
                else:
                    if not any(marker_positions[i]):  # 해당 리스트에 값이 없는 경우
                        print("Error 100: 6개의 board aruco marker가 인식되지 않는 경우")
                        # 초기화 작업
                        initial_count = 0
                        for j in range(7):
                            marker_positions[j].clear()
                        board_position.clear()
                        break
                    if marker_positions[i][0][2] <= 53:         # board 위의 aruco marker의 height(크기)로 구별함
                        print("Error 400: 다른 aruco marker가 board aruco marker를 대신해서 저장된 경우")
                        # 초기화 작업
                        initial_count = 0
                        for j in range(7):
                            marker_positions[j].clear()
                        board_position.clear()
                        break
                    else:
                        board_position[i] = marker_positions[i][0]   # 초기 화면에는 board에 햄스터와 주사위는 없다고 가정함.

            # # 결과 출력
            # #print("marker_positions:", marker_positions)  # 2차원 리스트
            # print("board_positions:", board_position)

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

    hamster = HamsterS()
    Ham = hamster_control()
    Ham.run()