#!/usr/bin/env python
from roslib import message
import rospy
import ros_numpy
from sensor_msgs.msg import PointField, PointCloud2,Image
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import LaserScan
from laser_geometry import LaserProjection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import cv_bridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

set1 = 0
set2 = 0
class cameraAnalysis():
    def __init__(self):
        self.imgSub = rospy.Subscriber("/cam1/image_raw", Image,self.imgCap)
        self.imgSub1 = rospy.Subscriber("/cam2/image_raw", Image,self.imgCap1)
        self.bridge = CvBridge()

    def imgCap(self, data):
        raw_input("Press Enter to save first img...")
        global set1
        try:
            savedIm = self.bridge.imgmsg_to_cv2(data, "bgr8")

            cv2.imwrite("/home/ros-dev-s434169/catkin_ws/non-essential-data/AnalysesImgs/img_setA%i.jpg" % set1, savedIm)
            set1=set1+1
            print 'saved image from first topic'
        except CvBridgeError as e:
            print(e)

    def imgCap1(self, data):
        raw_input("Press Enter to save second img...")
        global set2
        try:
            savedIm = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imwrite("/home/ros-dev-s434169/catkin_ws/non-essential-data/AnalysesImgs/img_setB%i.jpg" % set2, savedIm)
            set2 = set2 + 1
            print 'saved image from second topic'
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    rospy.init_node("depth2pc2")
    cA = cameraAnalysis()
    rospy.spin()
    

'''


maxim = 0
        for i in cv_image_16bit_encoding:
            if max(i[:])>maxim:
                maxim = max(i[:])
        print maxim#,minim
        factorIm = float(255)/maxim
        convertedImg = np.empty([480,640])
        for rw in range(0,479):
            for cl in range(0,639):
                convertedImg[rw][cl] = int(cv_image_16bit_encoding[rw][cl]*factorIm)
        convertedImg=cv2.cvtColor(convertedImg, cv2.CV_8U)
        im_color = cv2.applyColorMap(convertedImg, cv2.COLORMAP_JET)
        cv2.imshow("itle",im_color)
        cv2.waitKey(0)

for i in cv_image_16bit_encoding:
            s = set(i[:])
            print s.shape
            minlst.append(min(s[0]))
            maxlst.append(max(s[s.__len__()-1]))
        print minlst,maxlst



#print max(cv_image_16bit_encoding)
        #print min(cv_image_16bit_encoding)
        #rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
        #imgplot = plt.imshow(cv_image_16bit_encoding, cmap='gnuplot_r', interpolation='nearest')
        #plt.colorbar()
        #arr = imgplot._A
        #print np.array(arr).shape
        #print imgplot
        #plt.show()
        #cv2.imshow("Image window", v)
        #cv2.waitKey(100)


Code
 (rows, cols) = cv_image.shape
        if cols > 60 and rows > 60:
            cv_image=cv2.circle(cv_image, (50, 50), 10, 255)



class cameraAnalysis():
    def __init__(self):
        self.laserSub = rospy.Subscriber("/camera/depth/points", PointCloud2,self.ptAnalysis)

    def ptAnalysis(self,data):
        #generate heat map from pointcloud 2
        raw_input("Press Enter to continue...")
        cloud=list(pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z")))
        for x in cloud:
            print x

#cv_image_8bit_encoding = self.bridge.imgmsg_to_cv2(data, "8UC1")
            #lt.imshow(cv_image_16bit_encoding, cmap='gnuplot', interpolation='nearest')
            #plt.colorbar()
            #plt.show()
            #im_color = cv2.applyColorMap(cv_image_8bit_encoding, cv2.COLORMAP_JET)
            #cv2.imshow("title", im_color)
            #cv2.waitKey()




class cameraAnalysis():
    def __init__(self):
        self.laserSub = rospy.Subscriber("/camera/depth/points", PointCloud2,self.ptAnalysis)

    def ptAnalysis(self,data):
        assert isinstance(data, PointCloud2)
        gen = pc2.read_points(data)
        print type(gen)
        for p in gen:
            print p
        #generate heat map from pointcloud 2
        raw_input("Press Enter to continue...")
        #a = np.random.random((16, 16))
        #plt.imshow(a, cmap='hot', interpolation='nearest')
        #plt.show()
        #data_out = pc2.read_points(data.data, field_names=None, skip_nans=False, uvs=[[data.width, data.height]])

class cameraAnalysis():
    def __init__(self):
        self.laserSub = rospy.Subscriber("/camera/depth/image_raw", Image,self.ptAnalysis)

    def ptAnalysis(self, data):
        raw_input("Press Enter to continue...")
        print data.data
        
# 'msg' as type CompressedImage
depth_fmt, compr_type = msg.format.split(';')
# remove white space
depth_fmt = depth_fmt.strip()
compr_type = compr_type.strip()
if compr_type != "compressedDepth":
    raise Exception("Compression type is not 'compressedDepth'."
                    "You probably subscribed to the wrong topic.")

# remove header from raw data
depth_header_size = 12
raw_data = msg.data[depth_header_size:]

depth_img_raw = cv2.imdecode(np.fromstring(raw_data, np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
if depth_img_raw is None:
    # probably wrong header size
    raise Exception("Could not decode compressed depth image."
                    "You may need to change 'depth_header_size'!")

if depth_fmt == "16UC1":
    # write raw image data
    cv2.imwrite(os.path.join(path_depth, "depth_" + str(msg.header.stamp) + ".png"), depth_img_raw)
elif depth_fmt == "32FC1":
    raw_header = msg.data[:depth_header_size]
    # header: int, float, float
    [compfmt, depthQuantA, depthQuantB] = struct.unpack('iff', raw_header)
    depth_img_scaled = depthQuantA / (depth_img_raw.astype(np.float32)-depthQuantB)
    # filter max values
    depth_img_scaled[depth_img_raw==0] = 0

    # depth_img_scaled provides distance in meters as f32
    # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
    depth_img_mm = (depth_img_scaled*1000).astype(np.uint16)
    cv2.imwrite(os.path.join(path_depth, "depth_" + str(msg.header.stamp) + ".png"), depth_img_mm)
else:
    raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")
      def point_cloud(self, depth):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.

        depth is a 2-D ndarray with shape (rows, cols) containing
        depths from 1 to 254 inclusive. The result is a 3-D array with
        shape (rows, cols, 3). Pixels with invalid depth in the input have
        NaN for the z-coordinate in the result.

        """
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 0) & (depth < 255)
        z = np.where(valid, depth / 256.0, np.nan)
        x = np.where(valid, z * (c - self.cx) / self.fx, 0)
        y = np.where(valid, z * (r - self.cy) / self.fy, 0)
        return np.dstack((x, y, z))      
        
   '''