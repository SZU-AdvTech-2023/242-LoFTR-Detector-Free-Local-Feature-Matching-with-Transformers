import os
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

from src.loftr import LoFTR, default_cfg

import ransac
import blend
import k_means
from typing import List, Tuple, Union

import matplotlib.pyplot as plt


def show_image(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).show()

def drawMatches(img_left, img_right, kps_left, kps_right):
    H, status = cv2.findHomography(kps_right, kps_left, cv2.RANSAC)
    h_left, w_left = img_left.shape[:2]
    image = np.zeros((max(h_left, h_right), w_left + w_right, 3), dtype='uint8')
   
    image[0:h_left, 0:w_left] = img_right
    image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))  # (w,h
    image[0:h_left, 0:w_left] = img_left

    max_w = 2 * w_left
    for i in range(w_left, max_w): 
        if np.max(image[:, i]) == 0:
            max_w = i
            break
    print(max_w)
    return image[:, :max_w]

class Stitcher:
    def __init__(self, image1: np.ndarray, image2: np.ndarray):
        self.image1 = image1
        self.image2 = image2
        self.M = np.eye(3)

        self.image = None

    def stich(self, p1s, p2s, show_result=True, show_match_point=False, use_new_match_method=False, use_partial=False,
              use_gauss_blend=True):
        self.image_points1, self.image_points2 = p1s, p2s

        if use_new_match_method:
            self.M = ransac.GeneticTransform(self.image_points1, self.image_points2).run()
        else:
            self.M, _ = cv2.findHomography(
                self.image_points1, self.image_points2, method=cv2.RANSAC)

        print("Good points and average distance: ", ransac.GeneticTransform.get_value(
            self.image_points1, self.image_points2, self.M))

        left, right, top, bottom = self.get_transformed_size()
        width = int(max(right, self.image2.shape[1]) - min(left, 0))
        height = int(max(bottom, self.image2.shape[0]) - min(top, 0))

        if width * height > 8000 * 5000:
            factor = width * height / (8000 * 5000)
            width = int(width / factor)
            height = int(height / factor)

        if use_partial:
            self.partial_transform()

        self.adjustM = np.array(
            [[1, 0, max(-left, 0)],  # 横向
             [0, 1, max(-top, 0)],  # 纵向
             [0, 0, 1]
             ], dtype=np.float64)
        self.M = np.dot(self.adjustM, self.M)
        transformed_1 = cv2.warpPerspective(
            self.image1, self.M, (width, height))
        transformed_2 = cv2.warpPerspective(
            self.image2, self.adjustM, (width, height))

        self.image = self.blend(transformed_1, transformed_2, use_gauss_blend=use_gauss_blend)

        if show_match_point:
            for point1, point2 in zip(self.image_points1, self.image_points2):
                point1 = self.get_transformed_position(tuple(point1))
                point1 = tuple(map(int, point1))
                point2 = self.get_transformed_position(tuple(point2), M=self.adjustM)
                point2 = tuple(map(int, point2))

                cv2.circle(self.image, point1, 10, (20, 20, 255), 5)
                cv2.circle(self.image, point2, 8, (20, 200, 20), 5)

    def blend(self, image1: np.ndarray, image2: np.ndarray, use_gauss_blend=True) -> np.ndarray:
        mask = self.generate_mask(image1, image2)
        print("Blending")
        if use_gauss_blend:
            result = blend.gaussian_blend(image1, image2, mask, mask_blend=10)
        else:
            result = blend.direct_blend(image1, image2, mask, mask_blend=0)

        return result

    def generate_mask(self, image1: np.ndarray, image2: np.ndarray):
        center1 = self.image1.shape[1] / 2, self.image1.shape[0] / 2
        center1 = self.get_transformed_position(center1)
        center2 = self.image2.shape[1] / 2, self.image2.shape[0] / 2
        center2 = self.get_transformed_position(center2, M=self.adjustM)
        x1, y1 = center1
        x2, y2 = center2

        def function(y, x, *z):
            return (y2 - y1) * y < -(x2 - x1) * (x - (x1 + x2) / 2) + (y2 - y1) * (y1 + y2) / 2

        mask = np.fromfunction(function, image1.shape)

        mask = np.logical_and(mask, np.logical_not(image2)) \
               + np.logical_and(mask, image1) \
               + np.logical_and(image1, np.logical_not(image2))

        return mask

    def get_transformed_size(self) -> Tuple[int, int, int, int]:
        conner_0 = (0, 0)  # x, y
        conner_1 = (self.image1.shape[1], 0)
        conner_2 = (self.image1.shape[1], self.image1.shape[0])
        conner_3 = (0, self.image1.shape[0])
        points = [conner_0, conner_1, conner_2, conner_3]

        top = min(map(lambda x: self.get_transformed_position(x)[1], points))
        bottom = max(
            map(lambda x: self.get_transformed_position(x)[1], points))
        left = min(map(lambda x: self.get_transformed_position(x)[0], points))
        right = max(map(lambda x: self.get_transformed_position(x)[0], points))

        return left, right, top, bottom

    def get_transformed_position(self, x: Union[float, Tuple[float, float]], y: float = None, M=None) -> Tuple[
        float, float]:
        if isinstance(x, tuple):
            x, y = x
        p = np.array([x, y, 1])[np.newaxis].T
        if M is not None:
            M = M
        else:
            M = self.M
        pa = np.dot(M, p)
        return pa[0, 0] / pa[2, 0], pa[1, 0] / pa[2, 0]

class loftrInfer(object):

    def __init__(self, model_path="weights/indoor_ds.ckpt"):
        self.matcher = LoFTR(config=default_cfg)  
        self.matcher.load_state_dict(torch.load(model_path)['state_dict'])  
        self.matcher = self.matcher.eval().cuda()  

    def _infer_run(self, img0_raw, img1_raw):

        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.  
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1} 

        # Inference with LoFTR and get prediction 开始推理
        with torch.no_grad():
            self.matcher(batch)  # 网络推理
            mkpts0 = batch['mkpts0_f'].cpu().numpy()  # (n,2) 0的结果 -特征点
            mkpts1 = batch['mkpts1_f'].cpu().numpy()  # (n,2) 1的结果 -特征点
            mconf = batch['mconf'].cpu().numpy()  # (n,)      置信度

        # 筛选，需要四个以上的匹配点才能得到单应性矩阵
        if mconf.shape[0] < 4:
            return False
        mconf = mconf[:, np.newaxis]  
        np_result = np.hstack((mkpts0, mkpts1, mconf))  
        print(np_result.shape)
        list_result = list(np_result)

        def key_(a):
            return a[-1]

        list_result.sort(key=key_, reverse=True)  
        np_result = np.array(list_result)

        print('匹配点数:', len(np_result))
        return np_result

    def _points_filter(self, np_result, lenth=200, use_kmeans=True):

        lenth = min(lenth, np_result.shape[0])  # 选最大200个置信度较大的点对
        if lenth < 4: lenth = 4

        mkpts0 = np_result[:lenth, :2].copy()
        mkpts1 = np_result[:lenth, 2:4].copy()

        if use_kmeans:
            use_mkpts0, use_mkpts1 = k_means.get_group_center(mkpts0, mkpts1)  # 聚类，并返回同一类最多元素的匹配点
            print("一共：", mkpts0.shape)
            print("筛选与聚类后:", use_mkpts0.shape)
            if use_mkpts0.shape[0] < 4:
                return mkpts0, mkpts1
            return use_mkpts0, use_mkpts1
        return mkpts0, mkpts1

    def _draw_matchs(self, img1, img2, p1s, p2s, mid_space=10, if_save=True):
        h, w = img1.shape[:2]
        show = cv2.resize(img1, (2 * w + mid_space, h))
        show.fill(0)
        show[:, :w] = img1.copy()
        show[:, w + 10:] = img2.copy()
        p1s = p1s.astype(np.int)
        p2s = p2s.astype(np.int)

        for i in range(p1s.shape[0]):
            p1 = tuple(p1s[i])
            p2 = (p2s[i][0] + w + mid_space, p2s[i][1])
            cv2.line(show, p1, p2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                     1)  # 画线

        img_array = np.array(show)

        # 将RGB颜色通道改为BGR
        img_array = img_array[:, :, ::-1]


        plt.title("draw matchs")
        plt.imshow(img_array)
        plt.savefig('show.png')
        plt.show()

        if if_save:
            cv2.imwrite('save.jpg', show)

    def run(self, img0_bgr, img1_bgr, lenth=200, use_kmeans=True, if_draw=True, if_save=False, stitch_method=0):

        img0_bgr = cv2.resize(img0_bgr, (640, 480)) 
        img1_bgr = cv2.resize(img1_bgr, (640, 480))

        img0_raw = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)  
        img1_raw = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)

        np_result = self._infer_run(img0_raw, img1_raw)  # 推理
        if np_result is False:
            print("特征点数量不够！！！")
            return False
        mkpts0, mkpts1 = self._points_filter(np_result, lenth=lenth, use_kmeans=use_kmeans)  # 特征点筛选
        if if_draw:  # 显示匹配点对
            self._draw_matchs(img0_bgr, img1_bgr, mkpts0, mkpts1, mid_space=10, if_save=if_save)
        if stitch_method == 0:
            stitcher = Stitcher(img0_bgr, img1_bgr)
            stitcher.stich(p1s=mkpts0, p2s=mkpts1, use_partial=False, use_new_match_method=1, use_gauss_blend=0)
            image = (stitcher.image).copy()
        else:
            image = drawMatches(img0_bgr, img1_bgr, mkpts0, mkpts1)
        return image


if __name__ == "__main__":
    testInfer = loftrInfer(model_path="weights/outdoor_ds.ckpt")
    img1_pth = "assets/mytest/2.jpg"
    img0_pth = "assets/mytest/1.jpg"
    img0_bgr = cv2.imread(img0_pth) 
    img1_bgr = cv2.imread(img1_pth)
    print(img0_bgr.size)

    result = testInfer.run(img0_bgr, img1_bgr, lenth=1000, use_kmeans=True, if_draw=True, if_save=False,
                           stitch_method=0)

    result = cv2.resize(result, (800, 700))

    img_result = np.array(result)

    img_result= img_result[:, :, ::-1]


    plt.title('result')
    plt.xticks([]) 
    plt.yticks([])  

    plt.imshow(img_result)
    plt.savefig('result.png')
    plt.show()
