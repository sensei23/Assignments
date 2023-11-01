import cv2 as cv
import os

class MyOpenCV:
    def __init__(self, vid_name: str) -> None:
        self.video_obj = cv.VideoCapture(vid_name)

    def render_video(self) -> None:
        
        cnt = 1
        try:
            os.mkdir('folder')
        except:
            pass
        try:
            while self.video_obj.isOpened():
                ret, frame = self.video_obj.read()
                if not ret:
                    #cant access video anymore
                    break
                
                cv.imwrite('folder/frame_%d.jpg'%(cnt), frame)
                cnt += 1
        
        except Exception as e:
            print(e)


def main() -> None:
    cv_obj = MyOpenCV('video.mp4')
    cv_obj.render_video()

if __name__ == '__main__':
    main()