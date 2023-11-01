import cv2 as cv
import os

class MyOpenCV:
    def __init__(self) -> None:
        self.video_obj = cv.VideoCapture(0)

    def render_camera(self) -> None:

        if not self.video_obj.isOpened():
            print('cant open camera')
            return
        
        cnt = 1
        try:
            os.mkdir('webcam_folder')
        except:
            pass
        try:
            while self.video_obj.isOpened():
                ret, frame = self.video_obj.read()
                if not ret:
                    #cant access video anymore
                    break
                cv.imshow('webcam', frame)
                key_pressed = cv.waitKey(1)
                if key_pressed == 13:
                    # Enter key pressed
                    # saving frame and displaying it 
                    # cv.imshow('window', frame)
                    # cv.waitKey(0)
                    cv.imwrite('webcam_folder/frame_%d.jpg'%(cnt), frame)
                    cnt += 1
                elif key_pressed == 27:
                    break
        
        except Exception as e:
            print(e)
        finally:
            cv.destroyAllWindows()

def main() -> None:
    cv_obj = MyOpenCV()
    cv_obj.render_camera()

if __name__ == '__main__':
    main()