import cv2
import numpy as np
import argparse
import time
from jetcamera.camera import CSICamera

# Features Mather
def create_flann_matcher(algorithm, trees):
    index_params = dict(algorithm=algorithm, trees=trees)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann

def filter_matches(matches, ratio):
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, help="The Image's Path of Objetct to be tracked.", required=True)
    parser.add_argument("--capture_width", type=int, default=1280)
    parser.add_argument("--capture_height", type=int, default=720)
    parser.add_argument("--display_width", type=int, default=960)
    parser.add_argument("--display_height", type=int, default=570)
    opt = parser.parse_args()
    return opt

# Homography
def get_ROI(good_matches, at_least_match, image, kp_image, frame, kp_frame):
    if len(good_matches) > at_least_match:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        match_mask = mask.ravel().tolist()

        h, w = image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w -1 , 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        return dst
    return None

def wirte_text(frame, text_list):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, txt in enumerate(text_list):
        cv2.putText(frame, txt, (10, 20 + i * 20 ), font, 0.4, (0, 255, 0), 1)

def draw_lines(frame, opt):
    width = opt.display_width
    height = opt.display_height
    cv2.line(frame, (0, height // 2), (width, height // 2), (0, 255, 0), 1)
    cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
    cv2.circle(frame, (width // 2, height // 2), 50, (0, 255, 0), 1)


def main():
    opt = create_parser()    
    camera = CSICamera(opt.capture_width, opt.capture_height, opt.display_width, opt.display_height)
    query_image = cv2.imread(opt.track, 0)

    # Features Detector
    orb = cv2.ORB_create()
    # sift = cv2.SIFT_create()
    kp_image, desc_image = orb.detectAndCompute(query_image, None)

    # Feature Matcher
    flann = create_flann_matcher(0, 5)

    # Create Trackbar
    cv2.namedWindow("Match View")
    cv2.createTrackbar("Flann Ratio", "Match View", 7, 10, lambda x: None)
    cv2.createTrackbar("ROI Accuracy", "Match View", 10, 20, lambda x: None) 

    # Text Area
    text_list = []
    text_list.append("Track Image         : " + opt.track)
    text_list.append("Keypoints Algorithm : ORB")
    text_list.append("Match Algorithm     : Flann")
    text_list.append("Camera Input        : CSI")
    text_list.append("Capture Size        : 1280x720")
    text_list.append("FPS                 : 0")
    

    while True:
        start_time = time.time()
        frame = camera.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None) 

        if desc_frame is not None and desc_image is not None:
            matches = flann.knnMatch(np.asarray(desc_image, np.float32), np.asarray(desc_frame, np.float32), k=2)
            ratio = 0.1 if cv2.getTrackbarPos("Flann Ratio", "Match View") == 0 else (cv2.getTrackbarPos("Flann Ratio", "Match View") / 10)
            good_matches = filter_matches(matches, ratio)
            match_image = cv2.drawMatches(query_image, kp_image, gray_frame, kp_frame, good_matches, None)

            # get ROI
            at_least_match = 5 if cv2.getTrackbarPos("ROI Accuracy", "Match View") < 5 else (cv2.getTrackbarPos("ROI Accuracy", "Match View"))
            roi = get_ROI(good_matches, at_least_match, query_image, kp_image, gray_frame, kp_frame)
            if roi is not None:
                track_image = cv2.polylines(frame, [np.int32(roi)], True, (255, 0, 0), 3)
            else:
                track_image = frame
        else:
            match_image = gray_frame
            track_image = frame

        time_elapsed = time.time() - start_time
        text_list[-1] = "FPS                 : %2d" % (1 // time_elapsed)
        wirte_text(track_image, text_list)
        draw_lines(track_image, opt)
        cv2.imshow("Match View", match_image)
        cv2.imshow("Track View", track_image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()