# -*- coding:utf-8 -*-
#!/usr/bin/python
#
# Copyright (c) 2017, Aixi Wang <aixi.wang@hotmail.com>
# 
# reference project: https://github.com/jasonlfunk/ocr-text-extraction
#
# Copyright (c) 2012, Jason Funk <jasonlfunk@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#=========================================================

import cv2
#from matplotlib import pyplot
import json
import os
import math
import numpy as np
import time

DEBUG = 0
img_x = 0
img_y = 0
contours = []
img = None

# Determine pixel intensity
# Apparently human eyes register colors differently.
# TVs use this formula to determine
# pixel intensity = 0.30R + 0.59G + 0.11B
def ii(xx, yy):
    global img, img_y, img_x
    if yy >= img_y or xx >= img_x:
        #print "pixel out of bounds ("+str(y)+","+str(x)+")"
        return 0
    pixel = img[yy][xx]
    return 0.30 * pixel[2] + 0.59 * pixel[1] + 0.11 * pixel[0]


# A quick test to check whether the contour is
# a connected shape
def connected(contour):
    first = contour[0][0]
    last = contour[len(contour) - 1][0]
    return abs(first[0] - last[0]) <= 1 and abs(first[1] - last[1]) <= 1


# Helper function to return a given contour
def c(index):
    global contours
    return contours[index]


# Count the number of real children
def count_children(index, h_, contour):
    # No children
    if h_[index][2] < 0:
        return 0
    else:
        #If the first child is a contour we care about
        # then count it, otherwise don't
        if keep(c(h_[index][2])):
            count = 1
        else:
            count = 0

            # Also count all of the child's siblings and their children
        count += count_siblings(h_[index][2], h_, contour, True)
        return count


# Quick check to test if the contour is a child
def is_child(index, h_):
    return get_parent(index, h_) > 0


# Get the first parent of the contour that we care about
def get_parent(index, h_):
    parent = h_[index][3]
    while not keep(c(parent)) and parent > 0:
        parent = h_[parent][3]

    return parent


# Count the number of relevant siblings of a contour
def count_siblings(index, h_, contour, inc_children=False):
    # Include the children if necessary
    if inc_children:
        count = count_children(index, h_, contour)
    else:
        count = 0

    # Look ahead
    p_ = h_[index][0]
    while p_ > 0:
        if keep(c(p_)):
            count += 1
        if inc_children:
            count += count_children(p_, h_, contour)
        p_ = h_[p_][0]

    # Look behind
    n = h_[index][1]
    while n > 0:
        if keep(c(n)):
            count += 1
        if inc_children:
            count += count_children(n, h_, contour)
        n = h_[n][1]
    return count


# Whether we care about this contour
def keep(contour):
    return keep_box(contour) and connected(contour)

#----------------------------
# keep_box
#----------------------------
# Whether we should keep the containing box of this
# contour based on it's shape
def keep_box(contour):
    xx, yy, w_, h_ = cv2.boundingRect(contour)

    # width and height need to be floats
    w_ *= 1.0
    h_ *= 1.0

    # Test it's shape - if it's too oblong or tall it's
    # probably not a real character
    if w_ / h_ < 0.1 or w_ / h_ > 10:
        if DEBUG:
            print "\t Rejected because of shape: (" + str(xx) + "," + str(yy) + "," + str(w_) + "," + str(h_) + ")" + \
                  str(w_ / h_)
        return False
    
    # check size of the box
    if ((w_ * h_) > ((img_x * img_y) / 5)) or ((w_ * h_) < 15):
        if DEBUG:
            print "\t Rejected because of size"
        return False

    return True

#----------------------------
# include_box
#----------------------------
def include_box(index, h_, contour):
    if DEBUG:
        print str(index) + ":"
        if is_child(index, h_):
            print "\tIs a child"
            print "\tparent " + str(get_parent(index, h_)) + " has " + str(
                count_children(get_parent(index, h_), h_, contour)) + " children"
            print "\thas " + str(count_children(index, h_, contour)) + " children"

    if is_child(index, h_) and count_children(get_parent(index, h_), h_, contour) <= 2:
        if DEBUG:
            print "\t skipping: is an interior to a letter"
        return False

    if count_children(index, h_, contour) > 2:
        if DEBUG:
            print "\t skipping, is a container of letters"
        return False

    if DEBUG:
        print "\t keeping"
    return True
    
#----------------------------
# split_f_and_b
#----------------------------
def split_f_and_b(f1,f2):
    global img_x,img_y
    global contours
    global img
    
    if not os.path.isfile(f1):
        print "No such file '%s'" % f1
        return -1,None

    # Load the image
    orig_img = cv2.imread(f1)

    # Add a border to the image for processing sake
    img = cv2.copyMakeBorder(orig_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT)

    # Calculate the width and height of the image
    img_y = len(img)
    img_x = len(img[0])

    if DEBUG:
        print "Image is " + str(len(img)) + "x" + str(len(img[0]))

    #Split out each channel
    blue, green, red = cv2.split(img)

    # Run canny edge detection on each channel
    blue_edges = cv2.Canny(blue, 200, 250)
    green_edges = cv2.Canny(green, 200, 250)
    red_edges = cv2.Canny(red, 200, 250)

    # Join edges back into image
    edges = blue_edges | green_edges | red_edges

    # Find the contours
    image, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    hierarchy = hierarchy[0]

    if DEBUG:
        processed = edges.copy()
        rejected = edges.copy()

    # These are the boxes that we are determining
    keepers = []

    # For each contour, find the bounding rectangle and decide
    # if it's one we care about
    for index_, contour_ in enumerate(contours):
        if DEBUG:
            print "Processing #%d" % index_

        x, y, w, h = cv2.boundingRect(contour_)

        # Check the contour and it's bounding box
        if keep(contour_) and include_box(index_, hierarchy, contour_):
            # It's a winner!
            keepers.append([contour_, [x, y, w, h]])
            if DEBUG:
                cv2.rectangle(processed, (x, y), (x + w, y + h), (100, 100, 100), 1)
                cv2.putText(processed, str(index_), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        else:
            if DEBUG:
                cv2.rectangle(rejected, (x, y), (x + w, y + h), (100, 100, 100), 1)
                cv2.putText(rejected, str(index_), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    # Make a white copy of our image
    new_image = edges.copy()
    new_image.fill(255)
    boxes = []

    # For each box, find the foreground and background intensities
    for index_, (contour_, box) in enumerate(keepers):

        # Find the average intensity of the edge pixels to
        # determine the foreground intensity
        fg_int = 0.0
        for p in contour_:
            fg_int += ii(p[0][0], p[0][1])

        fg_int /= len(contour_)
        if DEBUG:
            print "FG Intensity for #%d = %d" % (index_, fg_int)

        # Find the intensity of three pixels going around the
        # outside of each corner of the bounding box to determine
        # the background intensity
        x_, y_, width, height = box
        bg_int = \
            [
                # bottom left corner 3 pixels
                ii(x_ - 1, y_ - 1),
                ii(x_ - 1, y_),
                ii(x_, y_ - 1),

                # bottom right corner 3 pixels
                ii(x_ + width + 1, y_ - 1),
                ii(x_ + width, y_ - 1),
                ii(x_ + width + 1, y_),

                # top left corner 3 pixels
                ii(x_ - 1, y_ + height + 1),
                ii(x_ - 1, y_ + height),
                ii(x_, y_ + height + 1),

                # top right corner 3 pixels
                ii(x_ + width + 1, y_ + height + 1),
                ii(x_ + width, y_ + height + 1),
                ii(x_ + width + 1, y_ + height)
            ]

        # Find the median of the background
        # pixels determined above
        bg_int = np.median(bg_int)

        if DEBUG:
            print "BG Intensity for #%d = %s" % (index_, repr(bg_int))

        # Determine if the box should be inverted
        if fg_int >= bg_int:
            fg = 255
            bg = 0
        else:
            fg = 0
            bg = 255

            # Loop through every pixel in the box and color the
            # pixel accordingly
        for x in range(x_, x_ + width):
            for y in range(y_, y_ + height):
                if y >= img_y or x >= img_x:
                    if DEBUG:
                        print "pixel out of bounds (%d,%d)" % (y, x)
                    continue
                if ii(x, y) > fg_int:
                    new_image[y][x] = bg
                else:
                    new_image[y][x] = fg

    # blur a bit to improve ocr accuracy
    new_image = cv2.blur(new_image, (2, 2))
    cv2.imwrite(f2, new_image)
    
    print 'saved new file:',f2
    if DEBUG:
        cv2.imwrite('edges.png', edges)
        cv2.imwrite('processed.png', processed)
        cv2.imwrite('rejected.png', rejected)
    return 0,new_image
    
#----------------------
# find_subimg_rect
#----------------------
def find_subimg_rect(img):
    h,w,n = img.shape
    
    print 'find_subimg_rect:','h:',h,' w:',w,' n:',n
    x_max = 0
    x_min = w-1
    y_max = 0
    y_min = h-1

    flag = 0
    for y in xrange(0,h):
        for x in xrange(0,w):               
            if img[y][x][0] == 0 and img[y][x][1] == 0 and img[y][x][2] == 255:
                #print y,x,img[y][x]
                flag = 1
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
                    
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
    
    print 'find_mask_rect:',x_min,y_min,x_max,y_max
    
    if flag == 1:
        return 0,[[x_min,y_min],[x_max,y_max]]
    else:
        return -1,[[x_min,y_min],[x_max,y_max]]
        

#----------------------
# find_angle
#----------------------
def find_angle(img):
    h,w,n = img.shape
    
    print 'find_angle:','h:',h,' w:',w,' n:',n
    x_max = 0
    x_min = w-1
    y_max = 0
    y_min = h-1
    
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    flag = 0
    for y in xrange(0,h):
        for x in xrange(0,w):               
            if img[y][x][0] == 0 and img[y][x][1] == 0 and img[y][x][2] == 255:
                #print y,x,img[y][x]
                flag = 1                   
                if x > x_max:
                    x_max = x
                    x2 = x
                    y2 = y
                    
                if x < x_min:
                    x_min = x
                    x1 = x
                    y1 = y
                    
    #print x_min,y_min,x_max,y_max
    
    if flag == 1:
        if x_max == x_min:
            angle = 90
        elif y_max == y_min:
            angle = 0
        else:
            l = math.sqrt((x1-x2)*(x1-x2) + (y2-y1)*(y2-y1))
            angle_hu = math.acos((x2-x1)/l)
            print 'angle_hu:',angle_hu
            angle = angle_hu*180.0/math.pi
        
        print 'angle:',angle
        
        return 0,angle
    else:
        return -1,0


#----------------------
# find_transform_points
#----------------------
def find_transform_points(img):
    h,w,n = img.shape
    
    points = []
    print 'find_transform_points:','h:',h,' w:',w,' n:',n
    x_max = 0
    x_min = w-1
    y_max = 0
    y_min = h-1
    
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    flag = 0
    for y in xrange(0,h):
        for x in xrange(0,w):               
            if img[y][x][0] == 0 and img[y][x][1] == 0 and img[y][x][2] == 255:
                #print y,x,img[y][x]
                flag = 1                                   
                points.append([x,y])
                    

    if flag == 1 and len(points) == 4:
        print points
        return 0,points
    else:
        return -1,[]
        

#----------------------
# find_segs_points
#----------------------
def find_segs_points(img):
    h,w,n = img.shape
    
    points = []
    print 'find_segs_points:','h:',h,' w:',w,' n:',n
    x_max = 0
    x_min = w-1
    y_max = 0
    y_min = h-1
    
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    flag = 0
    for y in xrange(0,h):
        for x in xrange(0,w):               
            if img[y][x][0] == 0 and img[y][x][1] == 0 and img[y][x][2] == 255:
                #print y,x,img[y][x]
                flag = 1                                   
                points.append([x,y])
                    

    if flag == 1:
        print points
        return 0,points
    else:
        return -1,[]




   
                    

    
#----------------------
# gen_config
#----------------------
def gen_config():
    mask_json = {}
    
    if os.path.exists('config.json'):
        try:
            f = open('config.json','rb')
            s = f.read()
            mask_json = json.loads(s)
            print 'read mask_json from file config.json'
            retcode = 0

        except:
            retcode = -1
        # return directly from config.json
        return retcode,mask_json

    #    
    # rebuild config.json
    #
    else:
        if os.path.exists('mask.bmp'):
            #retcode2,angle = find_angle(mask_img)
            mask_img = cv2.imread('mask.bmp')
            retcode1,transform_src = find_transform_points(mask_img)
        else:
            transform_src = []
            retcode1 = -1
            
        if os.path.exists('mask2.bmp'):
            mask_img = cv2.imread('mask2.bmp')
            retcode2,subimg_rect = find_subimg_rect(mask_img)
        else:
            retcode2 = -1
            subimg_rect = []
            
        if os.path.exists('mask3.bmp'):
            mask_img = cv2.imread('mask3.bmp')
            retcode3,seg_points = find_segs_points(mask_img)
        else:
            retcode3 = -1
            seg_points = []

        # get existed transform_des
        if os.path.exists('config2.json'):
            try:
                f = open('config2.json','rb')
                s = f.read()
                config2 = json.loads(s)
                transform_des = config2['transform_des']
                resize = config2['resize']
                rotate_base = config2['rotate_base']
                duration = config2['duration']
                seg_group_map = config2['seg_group_map']
                run_mode = config2['run_mode']
                print 'read mask_json from file config.json'

            except:
                transform_des = []
                resize = []
                rotate_base = 180
                duration = 60
                seg_group_map = []
                run_mode = 'normal'
        else:
            transform_des = []
            resize = []
            rotate_base = 180
            duration = 60
            seg_group_map = []
            run_mode = 'normal'
        retcode = 0

        mask_json = {
                     'subimg_rect':subimg_rect,
                     'seg_points':seg_points,
                     'resize':resize, 
                     'rotate':0,
                     'rotate_base':rotate_base,
                     'transform_src':transform_src,
                     'transform_des':transform_des,
                     'duration':duration,
                     'seg_group_map':seg_group_map,
                     'run_mode':run_mode
                    }
        mask_json_s = json.dumps(mask_json)
        f = open('config.json','wb')
        f.write(mask_json_s)
        f.close()
        print 'config.json generated'        
        return retcode,mask_json

#----------------------
# rotate
#----------------------
def rotate(img,angle):
    height = img.shape[0]
    width = img.shape[1]
    scale = 1
    rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
    rotateImg = cv2.warpAffine(img, rotateMat, (width, height))
    #cv2.imshow('rotateImg',rotateImg)
    #cv2.waitKey(0)
    return rotateImg #rotated image

#----------------------
# transform
#----------------------
def transform(img,points_src,points_des):
    height = img.shape[0]
    width = img.shape[1]
    scale = 1
    print 'transform:',height,width
        
    pts1 = np.float32(points_src)
    pts2 = np.float32(points_des)
    M = cv2.getPerspectiveTransform(pts1,pts2)    
    dst = cv2.warpPerspective(img,M,(width,height))
    return dst
    
#----------------------
# my_img_binarization
#---------------------- 
def my_img_binarization(f1,f2):
    img = cv2.imread(f1)
    new_image = img.copy()
    new_image.fill(255)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = im_gray.shape[0]
    w = im_gray.shape[1]
    sum = 0
    for y in xrange(0,h):
        for x in xrange(0,w):
            sum += im_gray[y][x]
            
    avg = sum/(h*w)       
    print 'avg:',avg

    # binarization
    for y in xrange(0,h):
        for x in xrange(0,w):
            if im_gray[y][x] < avg:
                new_image[y][x][2] = 0
                new_image[y][x][1] = 0
                new_image[y][x][0] = 0


    # remove isolated points
    for y in xrange(0,h):
        for x in xrange(0,w):
            if new_image[y][x][0] == 0 and (y == 0 or y == (h-1)):
                new_image[y][x][2] = 255
                new_image[y][x][1] = 255
                new_image[y][x][0] = 255
                
            elif new_image[y][x][0] == 0 and (x == 0 or x == (w-1)):
                new_image[y][x][2] = 255
                new_image[y][x][1] = 255
                new_image[y][x][0] = 255
            elif new_image[y][x][0] == 0 and new_image[y-1][x-1][0] == 255 and new_image[y+1][x+1][0] == 255 and new_image[y-1][x][0] == 255 and new_image[y+1][x][0] == 255 and new_image[y][x-1][0] == 255 and new_image[y][x+1][0] == 255 and new_image[y-1][x+1][0] == 255 and new_image[y+1][x-1][0] == 255:
                new_image[y][x][2] = 255
                new_image[y][x][1] = 255
                new_image[y][x][0] = 255
            else:
                pass

    cv2.imwrite(f2,new_image)
    return 0, new_image

#----------------------
# my_img_binarization_2
#---------------------- 
def my_img_binarization_2(f1,f2,ratio=0.8):

    print 'my_img_binarization_2 ratio:',ratio
    img = cv2.imread(f1)
    new_image = img.copy()
    new_image.fill(255)
    #imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
    h = img.shape[0]
    w = img.shape[1]
    
    sum = 0
    for y in xrange(0,h):
        for x in xrange(0,w):
            sum += imgYCC[y][x][0]
            #sum += imgGRAY[y][x]
            
    avg = sum/(h*w)       
    print 'avg:',avg

    # binarization
    for y in xrange(0,h):
        for x in xrange(0,w):
            if imgYCC[y][x][0] < avg*ratio:
            #if imgGRAY[y][x] < avg*0.8:
                new_image[y][x][2] = 0
                new_image[y][x][1] = 0
                new_image[y][x][0] = 0


    # remove isolated points
    for y in xrange(0,h):
        for x in xrange(0,w):
            if new_image[y][x][0] == 0 and (y == 0 or y == (h-1)):
                new_image[y][x][2] = 255
                new_image[y][x][1] = 255
                new_image[y][x][0] = 255
                
            elif new_image[y][x][0] == 0 and (x == 0 or x == (w-1)):
                new_image[y][x][2] = 255
                new_image[y][x][1] = 255
                new_image[y][x][0] = 255
            elif new_image[y][x][0] == 0 and new_image[y-1][x-1][0] == 255 and new_image[y+1][x+1][0] == 255 and new_image[y-1][x][0] == 255 and new_image[y+1][x][0] == 255 and new_image[y][x-1][0] == 255 and new_image[y][x+1][0] == 255 and new_image[y-1][x+1][0] == 255 and new_image[y+1][x-1][0] == 255:
                new_image[y][x][2] = 255
                new_image[y][x][1] = 255
                new_image[y][x][0] = 255
            else:
                pass

    cv2.imwrite(f2,new_image)
    return 0, new_image
    
#-------------------
# log_dump
#-------------------
def log_dump(filename,content):
    fpath = filename
    f = open(filename,'ab')
    t_s = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    
    content = t_s + '->' + str(content) + '\r\n'
    
    if type(content) == str:
        content_bytes = content.encode('utf-8')
        f.write(content_bytes)
    else:
        f.write(content)
    
    f.close()

#----------------------------
# gen_filename
#----------------------------        
def gen_filename(f_template,t):
    t_s = time.strftime('%Y%m%d-%H%M%S', time.localtime(t))
    return f_template.replace('%%',t_s)
    
#----------------------
# main
#----------------------
if __name__ == "__main__":
    pass
