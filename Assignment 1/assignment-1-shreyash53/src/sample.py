import cv2
import matplotlib.pyplot as plt
import numpy as np

class TestClass():
    def __init__(self):
        self.fname = '../images/Section-1/calib-object.jpg'
        self.img = plt.imread(self.fname)
        self.point = ()

    def getCoord(self):
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        plt.imshow(self.img)
        plt.show()
        # cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        # plt.show()
        return self.point

    def __onclick__(self,click):
        self.point = (click.xdata,click.ydata)
        return self.point


TestClass().getCoord()
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# # import mpl_toolkits.mplot3d

# ig = mpimg.imread('../images/Section-1/calib-object.jpg')
# # x = np.linspace(0, ig.shape[1], ig.shape[1]) #List of discrete x values
# x = [1527.49620235, 1520.5488272 , 1455.43674411, 1622.29395495,
#        1464.16943638, 1738.2031375 , 1359.36222703, 1892.91927019,
#        1241.47687408, 1732.90420387, 1096.15702455, 2070.22024634,
#         773.087833  , 2308.401887  , 1224.86704261, 1883.15967758,
#         773.087833  , 2069.64685359, 1023.14425607, 2068.69837464]
# # y = np.linspace(0, ig.shape[0], ig.shape[0]) #List of discrete y values
# y = [1667.31787759, 1775.20199078, 1782.80661796, 1783.62654194,
#        1661.11775714, 1925.83710958, 1936.78397542, 1664.01918984,
#        1975.7888179 , 2062.06380756, 1824.76792824, 1984.96288975,
#        2398.25471097, 2221.48319712, 2152.33762812, 2103.68627903,
#        2398.25471097, 2332.48995597, 2475.63606931, 2907.35266308]

# X, Y = np.meshgrid(x, y)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# #Plot the wireframe
# #I want to plot the image as f(x,y) and I can't understand why wireframe won't let me

# ax.plot_wireframe(X, Y, ig[:,:,2], rstride=10, cstride=10)

# plt.show()