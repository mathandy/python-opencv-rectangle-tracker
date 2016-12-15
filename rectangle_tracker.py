from operator import itemgetter
from math import ceil, floor, acos
import numpy as np
from numpy.linalg import norm
import cv2

video_file_location = 'sample.avi'
MAX_TIME_DELTA = 0.25  # for trackr
MIN_TIME_DELTA = 0.05  # for trackr
max_duration_to_track_in_mhi = 10  # seconds
frame_width, frame_height = 640, 424
fps = 24
rot180 = True  # If paper is upside down, change this
tol_corner_movement = 0.1
obst_tol = 2


def argmin(somelist):
    return min(enumerate(somelist), key=itemgetter(1))


def rotate180(src):
    (h, w) = src.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    return cv2.warpAffine(src, M, (w, h))


def persTransform(pts, H):
    src = np.zeros((len(pts), 1, 2))
    src[:, 0] = pts
    dst = cv2.perspectiveTransform(src, H)
    return np.array(dst[:, 0, :], dtype='float32')


def affTransform(pts, A):
    src = np.zeros((len(pts), 1, 2))
    src[:, 0] = pts
    dst = cv2.transform(src, A)
    return np.array(dst[:, 0, :], dtype='float32')


def unsharpMask(im, win, amt, k=1.5):
    """Perform in-place unsharp masking operation.
    win : should be 2-tuple of odd positive integers and amt a scalar"""
    tmp = cv2.GaussianBlur(im, win, amt)
    return cv2.addWeighted(im, k, tmp, -0.5, 0)


def dimg(im, nodes=None, node_colors=None, dontAlterInputImage=True,
         polygon=False, title='tmp', pause=False):
    if dontAlterInputImage:
        im2 = im.copy()
    else:
        im2 = im
    if nodes is not None:
        N = len(nodes)
        tnodes = [tuple(c) for c in nodes]
        if node_colors is None:
            node_colors = [(255, 0, 0)] * N
        for i in range(N):
            startpt = tnodes[(i - 1) % N]
            endpt = tnodes[i]
            cv2.circle(im2, startpt, 3, node_colors[(i - 1) % N], -1)
            if polygon:
                cv2.line(im2, startpt, endpt, (0, 0, 255), 2)
    cv2.imshow(title, im2)
    # Note: the `0xFF == ord('q')`is apparently necessary for 64bit machines
    if pause and cv2.waitKey(0) & 0xFF == ord('q'):
        pass


def run_main():
    # Initialize some variables
    old_homog = None
    old_inv_homog = None
    old_silhouette = None
    #    segIsGood = False
    corner_history = []
    old_dims = []
    mhi = np.float32(np.zeros((frame_height, frame_width)))

    cap = cv2.VideoCapture(video_file_location)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, frame_height)

    frame_count = 0
    while (True):
        # initialize some stuff
        c_colors = [(0, 0, 255)] * 4

        # grab current frame from video feed
        ret, frame = cap.read()
        roi = frame
        if not (frame_count % 10):
            print(frame_count)
        frame_count += 1

        # make copies and preprocessed versions of the current frame
        try:
            roi3 = roi.copy()
        except:
            print("\nVideo feed ended.\n")
            break

        # Convert to grayscale
        gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # get binary thresholding of image
        smgray_smth = cv2.GaussianBlur(gray_img, (15, 15), 0)
        _, bin_img = cv2.threshold(smgray_smth, 100, 255, cv2.THRESH_BINARY)

        # morphological closing
        kernel = np.ones((3, 3), np.uint8)
        closed_bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel,
                                   iterations=4)

        # Find corners.  To do this:
        # 1) Find the largest (area) contour in thresholdeded ,
        # 2) get contours convex hull,
        # 3) reduce degree of convex hull with Douglas-Peucker algorithm,
        # 4) refine corners with subpixel corner finder

        # step 1
        contours, _ = cv2.findContours(closed_bin_img,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv2.contourArea)

        # step 2
        hull = cv2.convexHull(biggest_contour)
        epsilon = 0.05 * cv2.arcLength(biggest_contour, True)

        # step 3
        hull = cv2.approxPolyDP(hull, epsilon, True)

        # step 4
        hull = np.float32(hull)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-4)
        cv2.cornerSubPix(gray_img, hull, (5, 5), (-1, -1), criteria)
        corners = [pt[0] for pt in hull]

        # Find top-right corner and use to label corners
        # Note: currently corners are in CW order
        tr_index = np.argmin(c[0] + c[1] for c in corners)
        tl = corners[tr_index]
        bl = corners[(tr_index - 1) % 4]
        br = corners[(tr_index - 2) % 4]
        tr = corners[(tr_index - 3) % 4]

        # reformat and ensure that ordering is as expected below
        corners = np.float32([[c[0], c[1]] for c in [tl, bl, br, tr]])

        # Check whether paper has moved since last from
        # Important assumptions on paper tracking (used in code block below):
        # 1) if any one point is stationary from previous frame, then all
        #    are stationary with probability 1.
        # 2) for now I can assume the paper will only experience movements
        #    on a solid planar surface, so (w.r.t said plane) each
        #    transformation should be of the form of a translation of one
        #    point (known) plus a rotation around that point (must be solved
        #    by a second point.
        last_unob_corners = None
        if corner_history:
            old_corners = corner_history[-1]

            def get_edge_lengths(topl, botl, botr, topr):
                """ Takes in list of four corners, returns four edge lengths
                in order top, right, bottom, left."""
                tbrl = [topr - topl, topr - botr, botr - botl, botl - topl]
                return [norm(edge) for edge in tbrl]

            # We'll for obstructions by looking for changes in edge lengths
            new_lengths = get_edge_lengths(*corners)
            if last_unob_corners is None:
                old_lengths = get_edge_lengths(*old_corners)
            else:
                old_lengths = get_edge_lengths(*last_unob_corners)

            top_is_bad, rgt_is_bad, bot_is_bad, lft_is_bad = \
                [abs(l0 - l1) > obst_tol for l1, l0 in zip(new_lengths, old_lengths)]
            tl_ob = top_is_bad and lft_is_bad
            bl_ob = bot_is_bad and lft_is_bad
            br_ob = bot_is_bad and rgt_is_bad
            tr_ob = top_is_bad and rgt_is_bad

            is_obstr = [tl_ob, bl_ob, br_ob, tr_ob]
            ob_indices = [i for i, c in enumerate(is_obstr) if c]
            ob_corner_ct = sum(is_obstr)
            c_colors = [(0, 255, 0) if b else (0, 0, 255) for b in is_obstr]

            # If one of the corners hasn't moved, then just assume none have
            diffs = [norm(corners[i] - old_corners[i]) for i in range(4)]
            min_idx, min_diff = argmin(diffs)
            if min_diff < tol_corner_movement:  # the paper hasn't moved
                for idx, d in enumerate(diffs):
                    if d > tol_corner_movement:
                        corners[idx] = old_corners[idx]

            else:  # The paper has moved, figure out where to
                if 0 < ob_corner_ct < 4:
                    print("frame={} | ob_corner_ct={}"
                          "".format(frame_count, ob_corner_ct))

                    old_good = np.float32([c for i, c in enumerate(old_corners) if not is_obstr[i]])
                    old_hidd = np.float32([c for i, c in enumerate(old_corners) if is_obstr[i]])
                    new_good = np.float32([c for i, c in enumerate(corners) if not is_obstr[i]])

                    p_old_good = persTransform(old_good, old_homog)
                    p_old_hidd = persTransform(old_hidd, old_homog)
                    p_new_good = persTransform(new_good, old_homog)

                # check for obstructions
                if ob_corner_ct == 0:  # yay! no obstructed corners!
                    last_unob_corners = corners

                elif ob_corner_ct == 1:
                    # Then use old_homog to transform old and new good points,
                    # then find affine transformation between them and map back
                    A = cv2.getAffineTransform(p_old_good, p_new_good)
                    p_new_hidd = affTransform(p_old_hidd, A)
                    new_hidd = persTransform(p_new_hidd, old_inv_homog)
                    corners[np.ix_(ob_indices)] = new_hidd

                elif ob_corner_ct == 2:
                    # Align the line between the good corners with the same
                    # line w.r.t the old corners
                    p1, q1 = p_new_good[0], p_new_good[1]
                    p0, q0 = p_old_good[0], p_old_good[1]
                    u0 = (q0 - p0) / norm(q0 - p0)
                    u1 = (q1 - p1) / norm(q1 - p1)
                    angle = acos(np.dot(u0, u1))  # unsigned
                    trans = p1 - p0

                    # Find rotation that moves u0 to u1
                    rotat = cv2.getRotationMatrix2D(tuple(p1), angle, 1)[:, :2]

                    # Expensive sign check for angle (could be improved)
                    if norm(np.dot(u0, rotat) - u1) > norm(np.dot(u1, rotat) - u0):
                        rotat = np.linalg.inv(rotat)

                    # transform the old coords of the hidden corners and map
                    # them back to desk plane
                    p_old_hidd += trans
                    p_new_hidd = affTransform(p_old_hidd, rotat)
                    new_hidd = persTransform(p_new_hidd, old_inv_homog)
                    corners[np.ix_(ob_indices)] = new_hidd

                elif ob_corner_ct == 3:
                    # Use the one good corner along with the vectors given by
                    # the paper's edges
                    p1 = p_new_good[0]
                    p0 = p_old_good[0]
                    trans = p1 - p0
                    print "Andy... you still need to fix me. frame = %s" % frame_count
                    angle = 0  ###MUST FIX

                    # Find rotation that moves u0 to u1
                    rotat = cv2.getRotationMatrix2D(p1, angle, 1)[:, 0:2]
                    if norm(np.dot(u0, rotat) - u1) > norm(np.dot(u1, rotat) - u0):
                        rotat = rotat ** (-1)

                    # Create and affine tranformation from `rotat` and `trans`
                    trans = np.array([trans]).T
                    A = np.hstack((rotat, trans))

                    # push transform the hidden old corner and map them back
                    # to desk plane
                    p_new_hidd = persTransform(p_old_hidd, A)
                    new_hidd = persTransform(p_new_hidd, old_inv_homog)
                    corners[np.ix_(ob_indices)] = new_hidd

                elif ob_corner_ct == 4:
                    # Note: should replace the crappy solution in this
                    # case with one based on shape matching.
                    print("Uh oh, all 4 corners obstructed... here's to"
                          "hoping that paper doesn't move.")
                    corners = old_corners
                else:
                    raise Exception("This should never happen.")

        # Now that the 4 corners are found:
        # update the mhi, get homographic transform, and display stuff
        w = max(abs(br[0] - bl[0]),
                abs(tr[0] - tl[0]))  # width of paper in pixels
        h = 11 * w / 8.5
        p_corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
        homog, mask = cv2.findHomography(corners, p_corners)
        inv_homog, inv_mask = cv2.findHomography(p_corners, corners)
        paper = cv2.warpPerspective(roi, homog, (int(ceil(w)), int(ceil(h))))
        if rot180:
            paper = rotate180(paper)

        p_silhouette = np.ones((int(floor(h)), int(floor(w))), dtype='float32')
        silhouette = cv2.warpPerspective(p_silhouette, inv_homog,
                                         (frame_width, frame_height))

        old_dims.append((w, h))
        corner_history.append(corners)
        old_homog = homog
        old_inv_homog = inv_homog
        if old_silhouette is None:
            old_silhouette = silhouette
        sil_mask = cv2.absdiff(old_silhouette, silhouette)
        sil_mask = np.array(sil_mask, dtype=np.result_type(bin_img))
        timestamp = float(frame_count) / fps
        cv2.updateMotionHistory(sil_mask, mhi, timestamp,
                                max_duration_to_track_in_mhi)
        old_silhouette = silhouette

        # display histogram of paper
        #        from andysmod import cv2hist
        #        cv2hist(paper)

        # Display images
        dimg(roi, corners, node_colors=c_colors, polygon=True,
             title='paper detected')
        dimg(paper, title="paper")

        # this is apparently necessary for 64bit machines
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_main()
    except:
        cv2.destroyAllWindows()
        raise
