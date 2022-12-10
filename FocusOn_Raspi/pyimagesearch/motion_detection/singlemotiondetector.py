import numpy as np
import imutils
import cv2

class smd:
    def __init__(self, accumWeight=0.5):
    #누적 가중치를 저장
        self.accumWeight = accumWeight

    #배경모델 초기화
        self.bg = None

    def update(self, image):
        #배경 모델이 None이면 초기화
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        #가중치를 누적하여 배경모델을 업데이트
        #평균
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def detect(self, image, tVal=25):
        #배경모델간의 절대 차이를 계산합니다.
        #전달된 이미지, 델타 이미지 임계값
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]

        #작은것을 제거하기 위해 일련의 침식과 팽창을 수행
        #blobs
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        #임계값 이미지에서 윤곽선을 찾고 초기화
        #모션을 위한 최소 및 최대 경계 상자 영역
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        #윤곽선이 밝견되지 않으면 None을 반환
        if len(cnts) == 0:
            return None

        #그렇지 않으면 등고선을 반복
        for c in cnts:
            #등고선의 경계 상자를 계산하고 이를 사용하여 
            #최소 및 최대 경계상자 영역을 업데이트
            (x, y, w, h) = cv2.boundingRect(c)
            (minX, minY) = (min(minX, x), min(minY, y))
            (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

        #그렇지 않으면 임계값 이미지의 튜플을 함께 반환
        #경계 상자 포함
        return (thresh, (minX, minY, maxX, maxY))
