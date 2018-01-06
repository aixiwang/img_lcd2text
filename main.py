# -*- coding:utf-8 -*-
#!/usr/bin/python
#
# Copyright (c) 2017, Aixi Wang <aixi.wang@hotmail.com>
#
#=========================================================

import cv2
import numpy as np
import sys
import os.path
import os

import time
import img_utils
import detect

try:
    import cap as cap
except:
    import cap_cv as cap

#if len(sys.argv) != 3:
#    print "%s input_file output_file" % (sys.argv[0])
#    sys.exit()
#else:
#    input_file = sys.argv[1]
#    output_file = sys.argv[2]#
#
#if not os.path.isfile(input_file):
#    print "No such file '%s'" % input_file
#    sys.exit()

INPUT_FILE = './img/in_%%.jpg'
OUTPUT_FILE = './img/out_%%.jpg'
P2_FILE = './img/P2_%%.jpg'
P3_FILE = './img/P3_%%.jpg'
P4_FILE = './img/P4_%%.jpg'
ERR_FILE = './img/err_%%.jpg'

TASK_DURATION = 60


    
#====================================
# main  
#====================================
if __name__ == "__main__":
    retcode,json_mask = img_utils.gen_config()

    while True:
        t1 = time.time()
        f1 = img_utils.gen_filename(INPUT_FILE,t1)
        f2 = img_utils.gen_filename(P2_FILE,t1)
        f3 = img_utils.gen_filename(P3_FILE,t1)
        f4 = img_utils.gen_filename(P4_FILE,t1)
        err_f = img_utils.gen_filename(ERR_FILE,t1)
        
        retcode,config_json = img_utils.gen_config()

        #
        # init local config values
        #
        print 'mask_jon:',config_json
        subimg_rect = config_json['subimg_rect']
        resize = config_json['resize']
        rotate = config_json['rotate']
        rotate_base = config_json['rotate_base']
        duration = config_json['duration']

        points_src = config_json['transform_src']
        points_des = config_json['transform_des']
        seg_points = config_json['seg_points']
        seg_group_map = config_json['seg_group_map']            
        run_mode = config_json['run_mode']
        
        # create folder for img storing
        if os.path.exists('./img') == False:
            os.system('mkdir img')

        # call cap function to capture picture
        print 'f1:',f1
        ret,w,h = cap.cap(f1,640,480)    
        if ret == -1:
            print 'capture image fail!'
        else:
            print 'f2:',f2
            try:
                if retcode == 0:

                    
                    if os.path.exists(f1):        
                        #img = cv2.imread(f2)
                        #          y y+h x x+w
                        raw_image = cv2.imread(f1)

                        if len(points_src) == 4 and len(points_des) == 4:
                            print 'step 1 =================> do transforming ...'
                            img2 = img_utils.transform(raw_image,points_src,points_des)

                        if (rotate_base + rotate) != 0:
                            print 'step 2 =================> do rotating ...'
                            img2 = img_utils.rotate(img2, (rotate_base + rotate)*(-1))
                        #else:
                        #    img2_new = img2

                        #if resize_x > 0 and resize_y > 0:
                        #    resized_image = cv2.resize(img2, (resize_x, resize_y))
                        #    cv2.imwrite(f2,resized_image)
                        #else:
                        #    cv2.imwrite(f2,img2)
                        cv2.imwrite(f2,img2)
                        
                        if len(subimg_rect) > 0:
                            print 'step 3 =================> do sub imaging ...'
                        
                            x1 = subimg_rect[0][0]
                            y1 = subimg_rect[0][1]
                            x2 = subimg_rect[1][0] 
                            y2 = subimg_rect[1][1] 
                            img3 = img2[y1:y2,x1:x2]
                            
                            cv2.imwrite(f3,img3)
                            
                            r = 1.0
                            while r > 0.8:
                                print 'r:',r
                                #ret,new_image = img_utils.split_f_and_b(f2,f3)
                                print 'step 4 =================> do binarization ...'
                                ret,new_image = img_utils.my_img_binarization_2(f3,f4,ratio=r)                               
                                
                                if len(seg_points)>0:
                                    print 'step 5 =================> do extract seg info. ...'
                                    if detect.do_detect(new_image,config_json) == 0:
                                        break
                                        
                                r -= 0.1
                                
                            if run_mode == 'normal':
                                try:
                                    os.remove(f2)
                                    os.remove(f3)
                                except:
                                    pass
                                

                        else:
                            print 'no subimg_rect defintion, passed reamaining steps'
                            pass
                        # call text recognition function here
                                                

                    else:
                        print 'no 1.jpg'
                else:
                    print 'no mask rectange found'            
                
                
                
            except Exception as e:
                print 'split_f_and_b exception',str(e)
                pass
            
        while (time.time() - t1) < duration:
            time.sleep(1)
