#!/usr/bin/python
#
# Copyright(c) 2017 Aixi Wang (aixi.wang@hotmail.com)
#
#==========================================================


#from PIL import Image
#import select
#import v4l2capture
import cv2

def cap(f,w,h):
    try:
    
        camera = cv2.VideoCapture(0)
        camera.set(3,w)
        camera.set(4,h)
        camera.set(5,30)

        retcode,image = camera.read()
        cv2.imwrite(f, image)
        size_y = image.shape[0]
        size_x = image.shape[1]
        del(camera)
        
        return 0,size_x,size_y
        
    except Exception as e:
        print 'exception:',str(e)
        return -1,0,0
