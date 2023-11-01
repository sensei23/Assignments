import cv2 as cv
import os

frame_size = (4096, 2160)

class MyOpenCV:
    def __init__(self, vid_name: str) -> None:
        self.video_obj = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(*'DIVX'), 45, frame_size)

    def render_frames(self) -> None:
        
        try:
            for c in range(len(os.listdir('folder/'))):
                img = cv.imread('folder/frame_%d.jpg'%(c+1), cv.IMREAD_UNCHANGED)
                self.video_obj.write(img)

            self.video_obj.release()
        except Exception as e:
            print(e)
        
def main() -> None:
    cv_obj = MyOpenCV('new_video1.avi')
    cv_obj.render_frames()

if __name__ == '__main__':
    main()
