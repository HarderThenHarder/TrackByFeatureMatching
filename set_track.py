from jetcamera.camera import CSICamera
import cv2
import numpy as np

start_pos = (0, 0)
end_pos = (0, 0)
pause = False
isMoving = False

def mouse_event(event, x, y, flags, param):
    global start_pos, end_pos, pause, isMoving
    if event == cv2.EVENT_LBUTTONDOWN:
        start_pos = (x, y)
        pause = True
        isMoving = True
    if event == cv2.EVENT_MOUSEMOVE:
        if isMoving:
            end_pos = (x, y)
    if event == cv2.EVENT_LBUTTONUP:
        isMoving = False

def main():
    camera = CSICamera()
    global start_pos, end_pos, pause

    cv2.namedWindow("Set Track")
    cv2.setMouseCallback("Set Track", mouse_event)

    i = 0
    while True:
        frame = camera.read()

        while pause:
            frame_cp = frame.copy()
            if end_pos != (0, 0):
                cv2.rectangle(frame_cp, start_pos, end_pos, (0, 255, 0), 2)
            cv2.putText(frame_cp, "Drag the mouse to rectangle the object, release the Button and press 's' to save image.", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
            cv2.imshow("Set Track", frame_cp)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit()
            elif key == ord('s') and pause:
                cv2.imwrite("%d_clip.jpg" % i, frame[start_pos[1] : end_pos[1], start_pos[0]: end_pos[0]])
                print("-> %d_clip.jpg has been saved!" % i)
                pause = False
                i += 1

        start_pos = (0, 0)
        end_pos = (0, 0)
        cv2.putText(frame, "Let the Camera look at the Object to be tracked, press the mouse left Button.", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
        cv2.imshow("Set Track", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()