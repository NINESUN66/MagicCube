import cv2
import numpy as np
import kociemba

# 定义图片展示函数
def cv_Show(name, img):
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_one_side_color(img): # 传入的图片需要已经裁切好

    # 将图片转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 设置红色的HSV范围
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # 创建一个掩码，将在指定范围内的颜色设为白色，其他颜色设为黑色
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 创建一个与原图大小相同的全黑图像
    result = np.zeros_like(img)

    # 将掩码中的白色部分复制到结果图像中
    result[mask != 0] = [255, 255, 255]

    # 将图像九等分，分别提取九等分的颜色，记录在color中
    color = [] # 存储九等分的颜色
    h, w = result.shape[:2]
    h1 = int(h/3)
    w1 = int(w/3)

    lower_orange = np.array([11, 100, 100])
    upper_orange = np.array([20, 255, 255])

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])

    lower_yellow = np.array([25, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    for i in range(3):
        for j in range(3):
            roi = result[i*h1:(i+1)*h1, j*w1:(j+1)*w1]

            center_x = i * h1 + h1 // 2
            center_y = j * w1 + w1 // 2
            center_color = hsv[center_x, center_y]

            
            if (lower_orange <= center_color).all() and (center_color <= upper_orange).all():
                color.append('L')
            elif (lower_red <= center_color).all() and (center_color <= upper_red).all():
                color.append('R')
            elif (lower_white <= center_color).all() and (center_color <= upper_white).all():
                color.append('U')
            elif (lower_yellow <= center_color).all() and (center_color <= upper_yellow).all():
                color.append('D')
            elif (lower_blue <= center_color).all() and (center_color <= upper_blue).all():
                color.append('B')
            elif (lower_green <= center_color).all() and (center_color <= upper_green).all():
                color.append('F')
            else:
                color.append('0')

    return color


img = cv2.imread('EntireMagic.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#展示轮廓
img_copy = img.copy()

# 遍历contours中的所有轮廓,找到周长大于1000的轮廓加入cnts中
cnts = []
for cnt in contours:
    if cv2.arcLength(cnt, True) > 200 and cv2.arcLength(cnt, True) < 1000:
        cnts.append(cnt)

# 根据轮廓切割图像并保存到数组中
img_list = []
for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    img_list.append(img[y:y+h, x:x+w])

# 整个魔方的数组
Entire = []
color = ['0','W','0','0','O','G','R','B','0','Y','0','0']

for img_temp in reversed(img_list):
    Entire.append(get_one_side_color(img_temp))

# 将Entire中的每个数据中的索引为[4]的元素按照U、R、F、D、L、B对Entire排序
# 自定义排序函数，根据指定的顺序排序
def custom_sort(element):
    order = "URFDLB"
    return order.index(element[4])

# 根据 [4] 索引的元素进行排序
Entire.sort(key=custom_sort)

# 按照白色、红色、绿色、黄色、橙色、蓝色的顺序存入string中
string = ''
for i in range(6):
    for j in range(9):
        string += Entire[i][j]

kociemba.solve(string)
