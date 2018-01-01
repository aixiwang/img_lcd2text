#!/usr/bin/python
#
# python-v4l2capture
#
# This file is an example on how to capture a picture with
# python-v4l2capture.
#
# 2009, 2010 Fredrik Portstrom
#
# I, the copyright holder of this file, hereby release it into the
# public domain. This applies worldwide. In case this is not legally
# possible: I grant anyone the right to use this work for any
# purpose, without any conditions, unless such conditions are
# required by law.

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
