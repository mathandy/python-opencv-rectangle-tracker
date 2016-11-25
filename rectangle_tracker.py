from itertools import product
from operator import itemgetter
import os
from math import ceil, floor, acos
import numpy as np
from numpy.linalg import norm
import cv2

video_file_location = 'sample.avi'
MAX_TIME_DELTA = 0.25  # for trackr
MIN_TIME_DELTA = 0.05  # for trackr
max_duration_to_track_in_mhi = 10  # seconds
frame_width,frame_height = 640, 424
fps = 24
rot180 = True  # If paper is upside down, change this
tol_corner_movement = 0.1
tol_edge_length = 5


def argmin(somelist):
    return min(enumerate(somelist), key=itemgetter(1))


def rotate180(src):
    (h, w) = src.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    return cv2.warpAffine(src, M, (w, h))


def persTransform(pts, H):
    src = np.zeros((len(pts),1,2))
    src[:,0] = pts
    dst = cv2.perspectiveTransform(src,H)
    return np.array(dst[:,0,:],dtype='float32')


def affTransform(pts, A):
    src = np.zeros((len(pts),1,2))
    src[:,0] = pts
    dst = cv2.transform(src,A)
    return np.array(dst[:,0,:],dtype='float32')


def unsharpMask(im, win, amt, k=1.5): 
    """Perform in-place unsharp masking operation.
    win : should be 2-tuple of odd positive integers and amt a scalar""" 
    tmp = cv2.GaussianBlur(im, win, amt)
    return cv2.addWeighted(im, k, tmp, -0.5, 0)


def dimg(im, nodes=[], node_colors=[], dontAlterInputImage=True, 
         polygon=False, title='tmp', pause=False):
    if dontAlterInputImage:
        im2 = im.copy()
    else:
        im2 = im
    if nodes!=[]:
        N = len(nodes)
        tnodes = [tuple(c) for c in nodes]
        if node_colors == []:
            node_colors = [(255,0,0)]*N
        for i in range(N):
            startpt = tnodes[(i-1)%N]; endpt = tnodes[i]
            cv2.circle(im2,startpt,3,node_colors[(i-1)%N],-1)
            if polygon:
                cv2.line(im2,startpt,endpt,(0,0,255),2)
    cv2.imshow(title,im2)
    #this weird syntax is apparently necessary for 64bit machines
    if pause and cv2.waitKey(0) & 0xFF == ord('q'):
        bla=1 #do nothing


def cornerVectors(corner, edges, grad, winSize=5):
    """INPUT:
    corner : 2x1 numpy array containing location of corner in image
    edges : binary image (the output of an edge detector)
    grad : grayscale image gradient (function)
    OUTPUT: [v1,v2] where v1 and v2 are 2x1 numpy arrays (the unit corner vectors)
    Note: Could speed up by finding edges and grad in here just for window"""
    q_app = np.reshape(round(corner),(1,2)) #pixel closest to q
    window = [np.arrya(shift)+ q_app for shift in product(range(winSize),range(winSize))]

    #Make list of all edge points in window
    edge_pts = [q_app + np.array([x,y]) for x,y in window if edges[x,y]!=0]

    #throw out any edge points that don't satisfy <(q-p),grad(p)>=0 (and thus aren't on paper's edges)
    edge_pts = [p for p in edge_pts if np.dot(q_app-p,grad(p))==0]

    #FINISH ME... !!!
    #fit two lines to edge_pts s.t. intersection is corner and output line unit vectors


def run_main():
    #Initialize some variables
    old_homog = None
    old_inv_homog = None
    old_silhouette = None
#    segIsGood = False
    old_corners = []
    old_dims = []
    mhi = np.float32(np.zeros((frame_height,frame_width)))

    cap = cv2.VideoCapture(video_file_location)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, frame_height)

    frame_count=0
    while(True):
        # initialize some stuff
        c_colors = [(0,0,255)]*4

        # grab current frame from video feed
        ret, frame = cap.read()
        roi = frame
        if not (frame_count % 10):
            print(frame_count)
        frame_count+=1

        # make copies and preprocessed versions of the current frame
        try:
            roi3 = roi.copy()
        except:
            print "\nVideo feed ended.\n"
            break
#            roi = unsharpMask(roi,(5,5),500,k=5)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
        grad = lambda i,j: np.array([sobelx[i,j]],sobely[i,j])

        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        gray_sharp = gray


        retval, thresho = cv2.threshold(gray_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Otsu's
        retval, thresh = cv2.threshold(gray_blur, 100, 255,cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel,iterations=4)

        cont_img = closing.copy()
#            cont_img = thresh.copy()
        contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        # Use countour to find Corners
        cnt = max([cnt for cnt in contours],key=cv2.contourArea)
        hull = cv2.convexHull(cnt)

        epsilon = 0.05*cv2.arcLength(cnt,True)
        hull2 = cv2.approxPolyDP(hull,epsilon,True)

        corners = np.float32(hull2)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,1000,0.0001)
        cv2.cornerSubPix(gray_sharp, corners, (5,5), (-1,-1), criteria)
        corners = [pt[0] for pt in corners]


        # label corners Note: currently they are in CW order
        tr_index = min(range(len(corners)),key = lambda i: corners[i][0] +
                        corners[i][1]) #top left corner index in corners
        tl = corners[tr_index]
        bl = corners[(tr_index-1)%4]
        br = corners[(tr_index-2)%4]
        tr = corners[(tr_index-3)%4]
#        corners = np.float32([[c[0],c[1]] for c in [tr,tl,bl,br]])
        corners = np.float32([[c[0],c[1]] for c in [tl,bl,br,tr]]) #<--order used below in ub_corners
        check_corner_colors = [(255,0,255),(0,255,0),(0,0,255),(0,0,0)]


#       Important assumptions on paper tracking (used in code block below):
#       1) if any one point is stationary from previous frame, then all
#          are stationary with probability 1.
#       2) for now I can assume the paper will only experience movements
#          on a solid planar surface, so (w.r.t said plane) each transformation should be of
#          the form of a translation of one point (known) plus a rotation
#          around that point (must be solved by a second point."""
        if old_corners != []:
            last_corners = old_corners[-1]

            # First find edge lengths to check for corner ubstructions
            top,right,bottom,left = [np.linalg.norm(v) for v in [tr-tl,tr-br,br-bl,bl-tl]] #paper edge lengths
            [otl,obl,obr,otr] = last_corners #<--see above corner order
            otop,oright,obottom,oleft = [np.linalg.norm(v) for v in [otr-otl,otr-obr,obr-obl,obl-otl]] #old paper edge lengths
            edge_diffs = [top-otop,right-oright,bottom-obottom,left-oleft]
            edge_ub = [(abs(d)>tol_edge_length).real for d in edge_diffs]
            tl_ub = edge_ub[3] and edge_ub[0]
            bl_ub = edge_ub[2] and edge_ub[3]
            br_ub = edge_ub[1] and edge_ub[2]
            tr_ub = edge_ub[0] and edge_ub[1]

            ub_corners = [tl_ub,bl_ub,br_ub,tr_ub] #<--see above corner order
            ub_indices = [i for i,c in enumerate(ub_corners) if c]
            ub_corner_ct = sum(ub_corners)
            # c_colors = [ifelse(c,(0,255,0),(0,0,255)) for c in ub_corners]
            c_colors = [(0,255,0) if c else (0,0,255) for c in ub_corners]

            # If one of the corners hasn't moved, then just assume none have
            diffs = [norm(corners[i]- last_corners[i]) for i in range(4)]
            min_idx, min_diff = argmin(diffs)
            if min_diff < tol_corner_movement: # the paper hasn't moved
                for idx, d in enumerate(diffs):
                    if d>tol_corner_movement:
                        corners[idx] = last_corners[idx]

            else:
            # The paper has moved, figure out where to

                if 0<ub_corner_ct<4:
#               compute some stuff used to find hidden corners
                    print "frame=%s | ub_corner_ct=%s"%(frame_count,ub_corner_ct)
                    old_good = np.float32([c for i,c in
                                enumerate(last_corners) if not ub_corners[i]])
                    old_hidd = np.float32([c for i,c in enumerate(last_corners) if ub_corners[i]])
                    new_good = np.float32([c for i,c in enumerate(corners) if not ub_corners[i]])

                    p_old_good = persTransform(old_good,old_homog)
                    p_old_hidd = persTransform(old_hidd,old_homog)
                    p_new_good = persTransform(new_good,old_homog)
                # check for ubstructions
                if ub_corner_ct == 1:
                # Then use old_homog to transform old and new good points,
                # then find affine transformation between them and map back
                    A = cv2.getAffineTransform(p_old_good,p_new_good)
                    p_new_hidd = affTransform(p_old_hidd,A)
                    new_hidd = persTransform(p_new_hidd,old_inv_homog)
#                    tmp = corners.copy()
                    corners[np.ix_(ub_indices)] = new_hidd
#                    print norm(tmp-corners)
#                    print "old_good = "
#                    print old_good
#                    print "new_hidd = "
#                    print new_hidd
#                    print "detected corners = "
#                    print tmp
#                    print "corners kept = "
#                    print corners


                elif ub_corner_ct == 2:
                    # Align the line between the good corners with the same 
                    # line w.r.t the old corners
                    p1,q1 = p_new_good[0],p_new_good[1]
                    p0,q0 = p_old_good[0],p_old_good[1]
                    u0 = (q0-p0)/norm(q0-p0)
                    u1 = (q1-p1)/norm(q1-p1)
                    angle = acos(np.dot(u0,u1)) # unsigned angle between old line and new line
                    trans = p1-p0 # moves p0 to p1

                    # Find rotation that moves u0 to u1
                    rotat = cv2.getRotationMatrix2D(tuple(p1), angle, 1)[:,0:2]
                    if norm(np.dot(u0,rotat) - u1) > norm(np.dot(u1,rotat) - u0):
                        rotat = np.linalg.inv(rotat)

#                    # Create and affine tranformation from [rotat; trans]
#                    trans = np.array([trans]).T
#                    A = np.hstack((rotat,trans))

                    # transform the old coords of the hidden corners and map 
                    # them back to desk plane
                    p_old_hidd += trans
                    p_new_hidd = affTransform(p_old_hidd,rotat)
                    new_hidd = persTransform(p_new_hidd,old_inv_homog)
                    corners[np.ix_(ub_indices)] = new_hidd
                elif ub_corner_ct == 3:
                    # Use the one good corner along with the vectors given by 
                    # the paper's edges
                    p1 = p_new_good[0]
                    p0 = p_old_good[0]
                    print "Andy... you still need to fix me. frame = %s"%frame_count
                    angle = 0 ###MUST FIX

                    # Find translation note: theoretically trans_p and trans_q 
                    # should be equal
                    trans_p = p1-p0 #moves p0 to p1
                    trans_q = q1-q0 #moves q0 to q1

                    #Find rotation that moves u0 to u1
                    rotat = cv2.getRotationMatrix2D(p1, angle, 1)[:,0:2]
                    if norm(np.dot(u0,rotat) - u1) > norm(np.dot(u1,rotat) - u0):
                        rotat = rotat**(-1)

                    # Create and affine tranformation from rotat & trans
                    trans = np.array([trans]).T
                    A = np.hstack((rotat,trans))

                    # push transform the hidden old corner and map them back 
                    # to desk plane
                    p_new_hidd = persTransform(p_old_hidd,A)
                    new_hidd = persTransform(p_new_hidd,old_inv_homog)
                    corners[np.ix_(ub_indices)] = new_hidd
                elif ub_corner_ct == 4:
#               Uh oh... should replace the crappy solution in this case with
#               one based on shape matching.
                    print("Uh oh, %s corners ubstructed"%ub_corner_ct)
#                    corners = np.float32([c+translation for c in last_corners]) #<--old solution
                    corners = last_corners
                elif ub_corner_ct > 4:
                    raise Exception("This should never happen.")
                else:
                    # yay! no ubstructed corners... do nothing
                    corners=corners

        # Now that the 4 corners are found, update the mhi, get homog, and 
        # display stuff
        w = max(abs(br[0]-bl[0]),abs(tr[0]-tl[0]))  # width of paper in pixels
        h = 11*w/8.5
        p_corners = np.float32([[0,0],[0,h],[w,h],[w,0]])
        homog, mask = cv2.findHomography(corners, p_corners)
        inv_homog, inv_mask = cv2.findHomography(p_corners, corners)
        paper = cv2.warpPerspective(roi, homog, (int(ceil(w)),int(ceil(h))))
        if rot180:
            paper = rotate180(paper)

        p_silhouette = np.ones((int(floor(h)), int(floor(w))), dtype='float32')
        silhouette = cv2.warpPerspective(p_silhouette, inv_homog, (frame_width,frame_height))
        gray_whiteout = np.uint8(gray*(1-silhouette))

        old_dims.append((w,h))
        old_corners.append(corners)
        old_homog = homog
        old_inv_homog = inv_homog
        if old_silhouette == None:
            old_silhouette = silhouette
        sil_mask = cv2.absdiff(old_silhouette,silhouette)
#        sil_mask = cv2.cvtColor(sil_mask, cv2.COLOR_BGR2GRAY)
        sil_mask = np.array(sil_mask,dtype=np.result_type(thresh))
        timestamp = float(frame_count) / fps
        cv2.updateMotionHistory(sil_mask, mhi, timestamp, max_duration_to_track_in_mhi)
        old_silhouette = silhouette

########trackr block (just some junk I was traying out)#############################
        # roi_trackr = roi.copy()
        # mgrad_mask, mgrad_orient = cv2.calcMotionGradient( mhi, MAX_TIME_DELTA,
        #                                                   MIN_TIME_DELTA, apertureSize=5 )
        # mseg_mask, mseg_bounds = cv2.segmentMotion(mhi, timestamp, MAX_TIME_DELTA)
        # for i, rect in enumerate([(0, 0, frame_width, frame_height)] + list(mseg_bounds)):
        #     x, y, rw, rh = rect
        #     area = rw * rh
        #     # TODO: where does 64**2 come from?
        #     if area < 64*2:
        #         continue
        #     motion_roi = mhi[y:y+rh, x:x+rw]
        #     if cv2.norm(motion_roi, cv2.NORM_L1) < 0.05 * area:
        #         # eliminate small things
        #         continue
        #     mgrad_orient_roi = mgrad_orient[y:y+rh, x:x+rw]
        #     mgrad_mask_roi = mgrad_mask[y:y+rh, x:x+rw]
        #     motion_hist_roi = mhi[y:y+rh, x:x+rw]
        #     angle = cv2.calcGlobalOrientation(mgrad_orient_roi, mgrad_mask_roi, motion_hist_roi,
        #                                       timestamp, max_duration_to_track_in_mhi)
        #     cv2.rectangle(roi_trackr, (x, y), (x+rw, y+rh), (0, 255, 0))
        #     cv2.putText(roi_trackr, "{:.1f}".format(angle), (x, y+rh), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
########end of trackr block##########################################################


        # Show good features to track in roi3
#        gf2track = cv2.goodFeaturesToTrack(gray,0,0.01,200)
#        for p in gf2track:
#            pt = tuple(p[0])
#            cv2.circle(roi3,pt,3,(255,0,0),-1)

        # Use Otsu's Method on just paper
        [paperB,paperG,paperR] = [chan for chan in cv2.split(paper)]
#        FINISH ME... use paperB for otsu's in below block
        paper_gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
        paper_gray_blur = cv2.GaussianBlur(paper_gray, (15, 15), 0)
        retval, paper_o = cv2.threshold(paper_gray_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        paper_oc = cv2.morphologyEx(paper_o, cv2.MORPH_CLOSE,kernel,iterations=1)

        # display histogram of paper
#        from andysmod import cv2hist
#        cv2hist(paper)

        # Display images
#        cv2.imshow("Morphological Closing", closing)
        dimg(roi,corners,node_colors=c_colors,polygon=True,title='paper detected')
        # dimg(roi,nodes=corners,node_colors=check_corner_colors,title='roi_colorcheck')
        # dimg(silhouette*thresh)
        # dimg(thresh,title="thresh")
        # dimg(paper_oc,title="Otsu closed")
        dimg(paper,title="paper")
#        cv2.imshow('trackr', roi)
#        cv2.imshow('just the paper', ifelse(rot180,rotate180(paper),paper))
#        cv2.imshow("gray_whiteout", gray_whiteout)
#        cv2.imshow("Motion History Image", mhi)
#        cv2.imshow("Adaptive Thresholding", thresh)
#        cv2.imshow('Contours', roi)
#        cv2.imshow('polygon', roi2)

        # this is apparently necessary for 64bit machines
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run_main()
    except:
        # bla = raw_input()
        cv2.destroyAllWindows()
        raise
