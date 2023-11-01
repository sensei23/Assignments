import cv2 as cv
import numpy as np

class MyOpenCV:
    def __init__(self) -> None:
        self.img1_obj = cv.imread('beach.jpeg', cv.IMREAD_UNCHANGED)
        self.img2_obj = cv.imread('mountain.jpeg', cv.IMREAD_UNCHANGED)

    def render_img(self) -> None:
        
        nw_img = np.array(self.img1_obj)
        nw_img[40:80, 80:460] = self.img2_obj[40:80, 80:460]
        nw_img[80:300, 250:290] = self.img2_obj[80:300, 250:290]

        self.display_image(nw_img)
        # cv.imwrite('output_image.jpg', nw_img)
        self.destroy()
    
    @staticmethod
    def display_image(img_obj) -> None:
        cv.imshow('window', img_obj)
        cv.waitKey(0)

    @staticmethod
    def destroy() -> None:
        cv.destroyAllWindows()


def main() -> None:
    cv_obj = MyOpenCV()
    cv_obj.render_img()


if __name__ == '__main__':
    main()