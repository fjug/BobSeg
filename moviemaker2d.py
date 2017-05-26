import cv2
import matplotlib.pyplot as plt
import numpy as np
import pylab
from matplotlib.path import Path


class MovieMaker2d:
    """
    Helps creating 2d+t movies showing segmentation results and flows with/without fiducial markers.
    """

    nihilation_radius = 15  # how big is the center area that kills fiducials
    do_respawn = False  # should killed fiducials respawn?
    respawn_margin = 20  # how many pixels from the cell outline should a killed fiucial respawn?

    def __init__( self, do_respawn ):
        """
        Parameters:
            do_respawn  -  fiducial markes that reach the center of the cell will be respawned near the membrane
        """
        self.do_respawn = do_respawn


    def render_flow_movie(self, data3d, ch0, ch1, folder=None, inline=False):
        '''
        Creates the frames of the flow movie and shows them. Optionally the frames can be saved directly.
        Parameters:
            data3d        -  a data3d object that already computed flow and segmentations
            ch0           -  a first raw data channel, images given as (t,y,x)
            ch1           -  a second raw data channel, images given as (t,y,x)
            folder        -  if not 'None' this folder will be used to store rendered frames.
            inline        -  False: openCV window will be used to display rendering
                             True: matplotlib is used (this can be used in jupyter inline mode!!!)
        '''
        assert ch0.shape[0] == ch1.shape[0]
        assert ch1.shape[0] == len(data3d.flows)

        if inline:
            from IPython.display import clear_output
            pylab.rcParams['figure.figsize'] = (25, 10)
            fig = plt.figure()

        hsv_shape = (ch0.shape[1], ch0.shape[2], 3)
        hsv = np.zeros(hsv_shape)
        hsv[..., 1] = 255

        # collect the fiducial dots
        dots = []
        for oid in range(0, len(data3d.object_names)):
            dots.extend(self.get_radialdots_in( data3d, 0, oid, 7, self.respawn_margin))  # self.get_griddots_in( data3d, 0,oid,spacing=15) )

        dot_history = [dots]
        for f in range(ch0.shape[0]):
            cells = []
            for oid in range(len(data3d.object_names)):
                if data3d.netsurf2dt is None:
                    cells.append(data3d.get_result_polygone(oid, f))
                else:
                    cells.append(data3d.get_result_polygone_2dt(oid, f))

            centers = []
            for oid in range(0, len(data3d.object_names)):
                centers.append(data3d.object_seedpoints[oid][f])

            outframe, dots = self.draw_flow(ch1[f],
                                            ch0[f],
                                            data3d.flows[f],
                                            dots=dots,
                                            dothist=dot_history,
                                            polygones=cells,
                                            show_flow_vectors=False,
                                            centers=centers,
                                            nihilation_radius=self.nihilation_radius)
            dot_history.insert(0, dots)

            # TESTCODE - DOT REPLACEMENT
            potentialnewdots = []
            for oid in range(0, len(data3d.object_names)):
                potentialnewdots.extend(self.get_radialdots_in( data3d, f, oid, 2, self.respawn_margin))
            for dot_index, dot in enumerate(dots):
                for oid in range(0, len(data3d.object_names)):
                    if np.linalg.norm(data3d.object_seedpoints[oid][f] - dot) < self.nihilation_radius:
                        if self.do_respawn:
                            dots[dot_index] = potentialnewdots[dot_index]
                        else:
                            dots[dot_index] = (-1, -1)
                        for dh in dot_history:
                            if self.do_respawn:
                                dh[dot_index] = potentialnewdots[dot_index]
                            else:
                                dh[dot_index] = (-1, -1)

            rgbframe = cv2.cvtColor(outframe, cv2.COLOR_BGR2RGB)

            # save frames if desired
            if not folder is None:
                cv2.imwrite(folder + 'frame%04d.png' % (f), outframe)

            if inline:
                pylab.axis('off')
                pylab.title("flow")
                pylab.imshow(rgbframe)
                pylab.show()
                clear_output(wait=True)
                # optional quick exit (DEBUGGING)
                if False and f == 3:
                    break
            else:
                cv2.imshow('flow', outframe)
                k = cv2.waitKey(25) & 0xff
                if k == 27:  # ESC
                    break

        if not inline:
            cv2.destroyAllWindows()

    def draw_flow(self, im, im2, flow,
                  step=16, dots=[], dothist=[], polygones=[], show_flow_vectors=True, centers=[], nihilation_radius=10):
        '''
        Renders an entire frame for flow movies. Lots of data needed here:
            im          -  flowchannel image for this frame
            im2         -  segchannel image for this frame (only needed to be visually complete)
            flow        -  the computed flow
            steps       -  number of pixel in between rendered flow arrows
            dots        -  position of dots to be first moved, then drawn
            dothist     -  old dots, so I can draw blue tails (list (time) of list (points) of (x,y) typles (positions))
            polygones   -  used to draw cell outlines
            show_flow_vectors - False: turn of rendering of grid spaced flow vectors (see 'steps')
        '''
        h, w = im.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
        fx, fy = flow[y, x].T

        # create image and draw
        vis = cv2.cvtColor(np.uint8(np.zeros_like(im)), cv2.COLOR_GRAY2BGR)
        im = im - np.min(im)
        im2 = im2 - np.min(im2)
        vis[:, :, 1] = (255 * im) / np.max(im)
        vis[:, :, 2] = (255 * im2) / np.max(im2)

        # flow arrows
        if (show_flow_vectors):
            lines = vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = int32(lines)

            for (x1, y1), (x2, y2) in lines:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 150, 150), 1)
                cv2.circle(vis, (x1, y1), 1, (0, 150, 150), -1)

        # draw polygones
        for polygone in polygones:
            cv2.polylines(vis, np.array([polygone], 'int32'), 1, (208, 224, 64), 2)

        # draw fiducial nihilation zone
        for (x, y) in centers:
            cv2.circle(vis, (int(x), int(y)), int(nihilation_radius), (148, 155, 85), 1)

        newdots = []
        for dot_idx, dot in enumerate(dots):
            # only show the dots that are 'alive' (dead ones are (-1,-1))
            if dot[0] == -1:
                continue

            # compute new dot
            try:
                fx, fy = flow[int(dot[1]), int(dot[0])].T
            except:
                fx = 0
                fy = 0
            newx = max(0, dot[0] + fx)
            newy = max(0, dot[1] + fy)
            newx = min(im.shape[1] - 1, newx)
            newy = min(im.shape[0] - 1, newy)
            newdot = (newx, newy)

            # history lines
            color = (255, 128, 128)
            intdot = (int(newdot[0]), int(newdot[1]))
            p1 = intdot
            for time_idx, histdots in enumerate(dothist):
                hdot = histdots[dot_idx]
                p2 = (int(hdot[0]), int(hdot[1]))
                if p1[0] != -1 and p2[0] != -1:  # don't show trail to 'dead' fiducials
                    cv2.line(vis, p1, p2, color, 1)
                color = tuple(np.array(color) - [10, 10, 10])
                if min(color) < 0:
                    break
                p1 = p2

            # point (after to draw on top of history lines etc)
            cv2.circle(vis, intdot, 3, (0, 165, 255), 1)

            newdots.append(newdot)

        return vis, np.array(newdots)

    def draw_segmentation(self, data3d, im, show_centers=True, dont_use_2dt=False, folder=None, inline=False):
        '''
        Draws movie frames for segmented objects with/without center points being visible.
            data3d        - data3d object that contains segmented objects
            im            -  the (raw?) image data to be rendered in the background
            show_centers  -  also center points will be draws (if True)
            dont_use_2dt  -  uses available 2dt data (if True; 2d data otherwise)
            folder        -  the folder to store the images in
            inline        -  if True it will show results within jupyter, otherwise in cv2 frame
        This method will return (frames, centers, polygones, radii), which is
            - a list of images (the frames of the created movie).
            - a list of (x,y)-tuples giving the found center points
            - a list of a list of polygones per frame (each polygone again given by a list of (x,y)-points)
            - a list of radii that denote the best fitting cirlcle (centered at the corresponding center point)
        '''
        frames = []
        centers = []
        radii = []
        all_polygones = []

        if inline:
            from IPython.display import clear_output
            pylab.rcParams['figure.figsize'] = (25, 10)
            fig = plt.figure()

        for f in range(len(im)):
            # create image for a single frame and draw
            vis = cv2.cvtColor(np.zeros_like(im[f]), cv2.COLOR_GRAY2BGR)
            vis[:, :, 0] = 255.0 * im[f] / np.max(im[f])
            vis[:, :, 1] = 255.0 * im[f] / np.max(im[f])
            vis[:, :, 2] = 255.0 * im[f] / np.max(im[f])

            # show center dot
            if show_centers:
                for oid in range(len(data3d.object_names)):
                    color = int(128 + 128. / len(data3d.object_names) * (oid + 1))
                    for f2 in range(f + 1):
                        center = tuple(data3d.object_seedpoints[oid][f2])
                        cv2.circle(vis, center, 3, (0, color, 0), 1)
                        centers.append(center)

                    # show best fitting circle
                    r = 0.
                    for i in range(data3d.num_columns):
                        r += data3d.netsurf2dt[oid].get_surface_index(f, i)
                    r /= data3d.num_columns
                    r /= data3d.K
                    r *= data3d.object_max_surf_dist[oid][f][0] - data3d.object_min_surf_dist[oid][f][0]
                    r += data3d.object_min_surf_dist[oid][f][0]
                    radii.append(r)

                    cv2.circle(vis, tuple(data3d.object_seedpoints[oid][f2]), int(r), (0, color, 0), 1)

            # retrieve polygones
            polygones = []
            for oid in range(len(data3d.object_names)):
                if data3d.netsurf2dt is None or dont_use_2dt:
                    polygones.append(data3d.get_result_polygone(oid, f))
                else:
                    polygones.append(data3d.get_result_polygone_2dt(oid, f))
            all_polygones.append(polygones)

            # draw polygones
            for polygone in polygones:
                cv2.polylines(vis, np.array([polygone], 'int32'), 1, (255, 0, 0), 2)

            rgbframe = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            frames.append(rgbframe)

            # save frames if desired
            if not folder is None:
                cv2.imwrite(folder + 'frame%4d.png' % (f), vis)

            if inline:
                pylab.axis('off')
                pylab.title("segmentation")
                pylab.imshow(rgbframe)
                pylab.show()
                clear_output(wait=True)
                # optional quick exit (DEBUGGING)
                if False and f == 3:
                    break
            else:
                cv2.imshow('segmentation', vis)
                k = cv2.waitKey(25) & 0xff
                if k == 27:  # ESC
                    break

        if not inline:
            cv2.destroyAllWindows()

        return frames, centers, all_polygones, radii

    def get_radialdots_in(self, data3d, frame, oid, spacing=5, pixels_inwards=5):
        '''
        Returns list of (x,y) coordinates placed at given distance inside a segmented polygon.
        Note that these points will only be placed if they where not shooting over the center.
            data3d           - data3d object that contains segmented objects
            frame            - guess
            oid              - object id
            spacing=5        - how many col-vectors to jump (5 === 20% of radial dots will be created)
            pixels_inwards=5 - how many pixels parallel (and inwards) from polygon will markers be placed?
        '''
        points = []

        netsurf = data3d.netsurfs[oid][frame]
        polypoints = np.array(data3d.get_result_polygone(oid, frame))

        cx = netsurf.center[0]
        cy = netsurf.center[1]
        for i in range(0, len(netsurf.col_vectors), spacing):
            x = polypoints[i, 0]
            y = polypoints[i, 1]
            dx = netsurf.col_vectors[i][0]
            dy = netsurf.col_vectors[i][1]
            x_new = int(x - dx * pixels_inwards)
            y_new = int(y - dy * pixels_inwards)
            # only if not shooting over center, add point!
            if np.sign(x - cx) - np.sign(x_new - cx) == 0:
                points.append((x_new, y_new))

        return points

    def get_griddots_in(self, data3d, frame, oid, spacing=5):
        '''
        Returns list of (x,y) coordinates placed at given distance inside a segmented polygon.
        Note that these points will only be placed if they where not shooting over the center.
            data3d           - data3d object that contains segmented objects
            frame            - guess
            oid              - object id
            spacing=5        - how many col-vectors to jump (5 === 20% of radial dots will be created)
        '''
        points = []

        polypoints = np.array(data3d.get_result_polygone(oid, frame))
        poly = Path(polypoints, closed=True)

        minx = np.min(polypoints[:, 0])
        maxx = np.max(polypoints[:, 0])
        miny = np.min(polypoints[:, 1])
        maxy = np.max(polypoints[:, 1])
        for x in range(minx, maxx, spacing):
            for y in range(miny, maxy, spacing):
                if poly.contains_point((x, y)):
                    points.append((x, y))

        return points