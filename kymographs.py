import numpy as np
import math
from skimage.filters import gaussian
from tifffile import imread, imsave
import cv2

import matplotlib.pyplot as plt
import pylab as pl

import bresenham as bham


class KymoSpider:
    """
    Implements Kymographs to visualize myesin and membrane flows
    """

    angles = None
    num_legs = None
    center = None
    length = None
    kymographs = None
    kymo_myosin = None
    kymo_seg = None
    kymo_flows = None
    kymo_membrane = None
    kymo_fiducials = None
    
    def __init__( self, length=100, center=(100,100), rotation=0 ):
        '''
        CONSTRUCTION
        length      - the length of the legs in pixels
        center      - (x,y)-tuple indicating the spider's center
        rotation    - rotation of the spieder (in degrees, counter-clockwise)
        '''
        self.num_legs = 8

        self.center = center  # the center point of the current spider
        self.set_leg_length(length)  # the length of each spider leg

        self.kymographs = [None] * self.num_legs  # the computed Kymographs for the membrane channel
        self.kymo_myosin = [None] * self.num_legs  # the computed Kymographs for the myosin channel
        self.kymo_seg = [None] * self.num_legs  # the computed Kymographs for the segmentation channel
        self.kymo_flows = [None] * self.num_legs  # the computed flow Kymographs for the flow

        self.membrane = [None] * self.num_legs  # the position of the membrane for each spider leg
        self.fiducials = [None] * self.num_legs  # the position of fiducials along spider legs

        self.set_leg_number(self.num_legs, rotation=rotation)

    def get_projected_length(self, vector, vec2project):
        '''
        Projects one vector onto another and returns the projected length.
        vector      - (x,y)-tuple to project on
        vec2project - (x,y)-tuple that should be projected onto given vector
        '''
        flow_x = vec2project[0]
        flow_y = vec2project[1]
        dx = vector[0]
        dy = vector[1]
        len_vector = math.sqrt(dx**2+dy**2)
        return (flow_x*dx + flow_y*dy)/len_vector
        #return flow_x+flow_y

    def get_pixel_values(self, image, coords):
        '''
        Returns a lit of pixel intensities of image at all given coords.
        image  - 2d image
        coords - list of (x,y)-coordinates to query the given image at
        '''
        values = []
        for c in coords:
            values.append(image[c[1]][c[0]])
        return values

    def move_fiducial(self, flow_kymo, init_positions):
        '''
        Moves a fiducial dot in a flow kymograph.
        flow_kymo      - a flow Kymograph (x-axis is time, y-axis is space, intensities are y-projected flow)
        init_positions - an integer defining the y-position of the fiducial dot in the flow Kymographs first column
        Returns a list of positions and a list of reset points (where fiducial moved out at bottom and a new one was created)
        '''
        pos = init_positions[0]
        print init_positions
        positions = [pos]
        resets = []
        for col in range(1,len(flow_kymo[0])): # 1 because flow at t==0 is blank
            print pos
            pos += flow_kymo[col, int(round(pos))]
            if pos>=len(flow_kymo)-1:
                pos = init_positions[col]
                resets.append((col,pos))
            positions.append(pos)
        return positions, resets
    
    def get_flow_stats(self, flow_kymo, start_idx, length=None, move_window=False):
        '''
        Computes the min,max,average (projected) and standard deviation of values along the given flow_kymo.
        t          -- time index in flow_kymo
        start_idx  -- index at which to start averaging (its a row idex in flow_kymo)
        length     -- number of pixels downwards from start_idx to take avg of
        '''
        avg = np.zeros(len(flow_kymo[0]))
        minimum = np.zeros_like(avg)
        maximum = np.zeros_like(avg)
        std = np.zeros_like(avg)
        resets = []
        
        for col in range(0,len(flow_kymo[0])):
            if start_idx[col]<0:
                start_idx[col] = 0
            end_idx = len(flow_kymo)
            if length is not None:
                end_idx = int( min(end_idx, start_idx[col]+length) )
            minimum[col] = np.min(flow_kymo.T[col,int(start_idx[col]):end_idx])
            maximum[col] = np.max(flow_kymo.T[col,int(start_idx[col]):end_idx])
            avg[col]     = np.average(flow_kymo.T[col,int(start_idx[col]):end_idx])
            std[col]     = np.std(flow_kymo.T[col,int(start_idx[col]):end_idx])
            # if desired, move window around by the measured flow
            if move_window and (col+1)<len(flow_kymo[0]):
                nextpos = start_idx[col] + avg[col]
                # respawn in case the window is pushed down so it would stick out at bottom
                if (nextpos+length) < len(flow_kymo):
                    start_idx[col+1] = nextpos
                else:
                    resets.append((col+1,start_idx[col+1]))

        return minimum, avg, maximum, std, start_idx, resets
    
    def set_center(self, x, y):
        '''
        Sets the center for the kymograph spider. (Given in pixels.)
        Return the set center tuple.
        '''
        self.center = (x,y)
        return self.center
    
    def get_center(self):
        '''
        Returns the current spider center coordinates.
        '''
        return self.center
    
    def set_leg_length(self, length):
        '''
        Sets the length of the individual Kymographs radiating out the center.
        '''
        assert(self.center[0]>=length) # spider not allowed to stick out of image
        assert(self.center[1]>=length) # spider not allowed to stick out of image

        self.length = length

    def set_leg_number(self, num_legs, rotation=0):
        '''
        :param num_legs: sets the given number of legs and distributes them uniformly around the center 
        :return: 
        '''
        self.num_legs = num_legs

        angles = np.arange(rotation, rotation + 360, 360.0 / self.num_legs)
        angles %= 360
        angles = np.rint(angles)
        self.set_leg_angles(angles)

    def set_leg_angles(self, leg_angles):
        '''
        leg_angles  - list of angles (in degrees) into which the  kymograph lines should point
        '''
        self.angles = leg_angles
        self.num_legs = len(self.angles)

    def get_leg_length(self):
        '''
        Returns the set leg length.
        '''
        return self.length
    
    def get_kymo_line_coordinates(self, p_start, p_end):
        '''
        Returns self.get_leg_length() many coordinates from p_start to p_end (along staight line).
        '''
        xs = np.linspace(p_start[0], p_end[0], self.length, dtype=np.dtype(int))
        ys = np.linspace(p_start[1], p_end[1], self.length, dtype=np.dtype(int))
        return zip(xs,ys)
    
    def get_leg_line(self, legnum):
        '''
        Returns a list if two (x,y)-coordinate tuples defining the start and endpoint of the desired spiderleg.
        '''
        assert legnum < len(self.angles)

        rad_rot=(self.angles[legnum]/360.0)*math.pi*2
        dx = int( math.sin(rad_rot)*self.length )
        dy = int( math.cos(rad_rot)*self.length )
        x = self.center[0]+dx
        y = self.center[1]+dy
        return [self.center, (x,y)]
    
    def get_leg_vector(self, legnum):
        '''
        Returns a 2d tuple containing a vector that defines the requested leg. Note that this vector will be centered
        at self.center!!!
        '''
        [c, p] = self.get_leg_line(legnum)
        return (p[0]-c[0], p[1]-c[1])
        
    def compute(self, img_membrane, img_myosin, img_seg, img_flow):
        '''
        Computes the Kymograph spider on the given image (located at the previously set center point and leg length).
        img_membrane  - the (t,y,x)-image data containing membrane label intensities
        img_myosin    - the (t,y,x)-image data containing myosin label intensities
        img_seg       - the (t,y,x)-image data containing the segmentation outlines only
        img_flow      - the (c,t,y,x)-image data containing the flow information (c[0]==x, c[1]==y)
        '''
        assert(np.shape(img_membrane)==np.shape(img_myosin))
        assert(np.shape(img_membrane)==np.shape(img_seg))
        assert(np.shape(img_flow)[0]==2)
        assert(np.shape(img_membrane)==np.shape(img_flow[0]))
        assert(np.shape(img_membrane)==np.shape(img_flow[1]))
        
        assert(np.shape(img_membrane)[1]-self.center[1] >= self.get_leg_length()) # spider must fit into image (y-dir.)
        # I made it easy for me by checking for a circle around the spider fitting the image...
        assert(np.shape(img_membrane)[2]-self.center[0] >= self.get_leg_length()) # spider must fit into image (x-dir.)
        
        for legnum in range(self.num_legs):
            [p1,p2] = self.get_leg_line(legnum)
            r = self.compute_leg_kymos(img_membrane, img_myosin, img_seg, img_flow, p2, p1)
            self.kymographs[legnum]  = r[0]
            self.kymo_myosin[legnum] = r[1]
            self.kymo_seg[legnum]    = r[2]
            self.kymo_flows[legnum]  = r[3]

    def compute_leg_kymos(self, img_membrane, img_myosin, img_seg, img_flow, p_start, p_end):
        '''
        Computes Kymographs et al for a leg between the two given coordinates.
        img_membrane  - the (t,y,x)-image data containing membrane label intensities
        img_myosin    - the (t,y,x)-image data containing myosin label intensities
        img_seg       - the (t,y,x)-image data containing the segmentation outlines only
        img_flow      - the (c,t,y,x)-image data containing the flow information (c[0]==x, c[1]==y)
        p_start       - tuple (x,y) pointing at the first pixel of the leg (kymo line)
        p_end         - tuble (x,y) pointing at the last pixel of the leg (kymo line)
        '''
        #line_pixels = bham.bresenhamline(np.array([p_start]),np.array([p_end]))
        line_pixels = self.get_kymo_line_coordinates(p_start,p_end)
                               
        kymo_membrane = np.zeros((len(line_pixels),len(img_membrane)))
        kymo_myosin = np.zeros((len(line_pixels),len(img_myosin)))
        kymo_seg = np.zeros((len(line_pixels),len(img_seg)))
        kymo_projflow = np.zeros((len(line_pixels),len(img_flow[0])))
                               
        for col in range(len(img_membrane)):
            fx = self.get_pixel_values(img_flow[0][col], line_pixels)
            fy = self.get_pixel_values(img_flow[1][col], line_pixels)
            
            for row in range(len(line_pixels)):
                x = line_pixels[row][0]
                y = line_pixels[row][1]

                # DEBUG: from IPython.core.debugger import Tracer; Tracer()()

                kymo_membrane[row,col] = img_membrane[col,y,x]
                kymo_myosin[row,col] = img_myosin[col,y,x]
                kymo_seg[row,col] = img_seg[col,y,x]
                vec = (p_end[0]-p_start[0], p_end[1]-p_start[1])
                kymo_projflow[row,col] = self.get_projected_length( vec, (fx[row],fy[row]))
        return (kymo_membrane, kymo_myosin, kymo_seg, kymo_projflow)
    
    def plot_spider_loc_on_images(self, fig, image, flow_image):
        '''
        Shows the current kymograph spider in the given figure.
        fig        - the figure object to plot into
        image      - the 2d+t image data containing intensities
        flow_image - the 2 channel 2d+t images containing the flow information (ch0==x, ch1==y)
        '''
        fig.suptitle('Sir, your spider!', fontsize=16)
        ax = fig.add_subplot(321)
        ax.imshow(image[0], plt.get_cmap('gray'))
        self.plot_spider_on_axis(ax)
        ax = fig.add_subplot(322)
        ax.imshow(image[-1], plt.get_cmap('gray'))
        self.plot_spider_on_axis(ax)
        
    def plot_spider_on_axis(self, ax):
        '''
        Allows you to hand and subplot Axis and you will get the spider plotted onto it.
        '''
        font1 = {'family': 'serif',
                 'color':  'yellow',
                 'weight': 'normal',
                 'size': 12,
                }

        font2 = {'family': 'serif',
                 'color':  'cyan',
                 'weight': 'normal',
                 'size': 12,
                }

        for legnum in range(self.num_legs):
            [p1,p2] = self.get_leg_line(legnum)
            if legnum == 0:
                ax.plot([p2[0],p1[0]],[p2[1],p1[1]],'y-',lw=2)
                ax.text(p2[0],p2[1],str(legnum+1),font1)
            else:
                ax.plot([p2[0],p1[0]],[p2[1],p1[1]],'c-',lw=2)
                ax.text(p2[0],p2[1],str(legnum+1),font2)

    def plot(self, fig, frame_first, frame_last, pos_fiducial, rel_to_membrane=True):
        '''
        Plots the computed Kymographs for the entire spider.
        fig             - the figure object to plot into
        frame_first     - (y,x)-image of first frame
        frame_last      - (y,x)-image of last frame
        pos_fiducial    - positioning of the initial fiducial marker (see rel_to_membrane for further info)
        rel_to_membrane - if True, pos_fiducial will be interpreted as a number relative to the segmented membrane pos
        '''
        fig.suptitle('Kymograph Spider', fontsize=16)
        
        # - - - PLOT OVERVIEWS - - - -
        ax = fig.add_subplot(2,self.num_legs+1,1)
        ax.imshow(frame_first)
        self.plot_spider_on_axis(ax)
        ax = fig.add_subplot(2,self.num_legs+1,self.num_legs+1+1)
        ax.imshow(frame_last)
        self.plot_spider_on_axis(ax)

        for legnum in range(self.num_legs):
            if legnum == 0:
                style = 'y.'
                style_reset = 'c*'
            else:
                style = 'c.'
                style_reset = 'y*'

            #from IPython.core.debugger import Tracer; Tracer()()
                        
            # compute all places where fiducials would be reinitiated IF they where to drop out at the bottom
            init_positions = [pos_fiducial] * len(self.kymographs[0][0])
            if rel_to_membrane:
                for i in range(len(self.kymographs[0][0])):
                    init_positions[i] += np.argmax(self.kymo_seg[legnum][:,i])
                    
            # make kymo_seg transparent (in order to be able to plot on top of another kymo)
            kymo_seg_transp = np.ma.masked_where(self.kymo_seg[legnum] < .9, self.kymo_seg[legnum])

            # - - - start the plotting - - - -
            
            ax = fig.add_subplot(2,self.num_legs+1,legnum+2)
            ax.imshow(self.kymographs[legnum], plt.get_cmap('gray'))
            # - - - - 
            ax.imshow(kymo_seg_transp, plt.get_cmap('Reds'), vmin=0, vmax=255, alpha=.9)
            # - - - -
            positions, resets = self.move_fiducial(self.kymo_flows[legnum],init_positions)
            if (len(resets)>0):
                ax.plot(zip(*resets)[0], zip(*resets)[1], style_reset, markersize=16)
            ax.plot(positions, style)
            #ax.axis('off')
            
            # - - - -
            
            ax = fig.add_subplot(2,self.num_legs+1,self.num_legs+1+legnum+2)
            #ax.imshow(self.kymo_flows[legnum], plt.get_cmap('gray'))
            ax.imshow(self.kymo_myosin[legnum], plt.get_cmap('gray'))
            # - - - -
            ax.imshow(kymo_seg_transp, plt.get_cmap('Reds'), vmin=0, vmax=300, alpha=.9)
            # - - - -
            positions, resets = self.move_fiducial(self.kymo_flows[legnum],init_positions)
            if (len(resets)>0):
                ax.plot(zip(*resets)[0], zip(*resets)[1], style_reset, markersize=16)
            ax.plot(positions, style)
            #ax.axis('off')

    def moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    def plot_column_flow_stats(self, fig, offset_from_membrane=0, length=None):
        '''
        Plots the several statistics computed per column of the computed kymographs.
        fig                   -  the figure object to plot into
        offset_from_membrane  -  stats to be computed in a stripe below membrane starting that many pixels below
        length                -  pixel height of stripe within which stats are computed (if None, stripe reaches until bottom)
        '''
        fig.suptitle('Flow Stats', fontsize=16)
        
        for legnum in range(self.num_legs):
            if legnum == 0:
                style = 'y.'
            else:
                style = 'c.'

            # read the segmented membrane location out of kymo_seg
            pos_membranes = np.zeros( len(self.kymographs[0][0]) )
            for col in range( len(self.kymographs[0][0]) ):
                pos_membranes[col] = np.argmax(self.kymo_seg[legnum][:,col])
            
            # get the average inward flow below the membrane
            #print np.shape(pos_membranes), pos_membranes
            minf, avgf, maxf, stdf, pos_intervals, resets = \
                    self.get_flow_stats(self.kymo_flows[legnum],pos_membranes+offset_from_membrane,length)
            
            # make kymo_seg transparent (in order to be able to plot on top of another kymo)
            kymo_seg_transp = np.ma.masked_where(self.kymo_seg[legnum] < .9, self.kymo_seg[legnum])

            # - - - start the plotting - - - -
                                     
            ax = fig.add_subplot(2,self.num_legs,legnum+1)
            ax.imshow(self.kymo_myosin[legnum], plt.get_cmap('gray'))
            ax.imshow(kymo_seg_transp, plt.get_cmap('Reds'), vmin=0, vmax=255, alpha=.9)
            ax.plot(pos_membranes+offset_from_membrane, style)
            if length is None:
                ax.plot(np.zeros_like(pos_membranes)+len(self.kymographs[0]), style)
            else:
                ax.plot(pos_membranes+offset_from_membrane+length, style)

            ax = fig.add_subplot(2,self.num_legs,self.num_legs+legnum+1)
            ax.set_ylim([-10,10])
            ax.plot(minf, color='gray')
            ax.plot(maxf, color='grey')
            ax.plot(avgf, 'b-')
            ax.plot(np.zeros_like(avgf), 'r-')
            ax.plot(self.moving_average(avgf, n=5), color='orange')

    def get_slippage_rates(self, flow_per_t, pos_membranes, delta_t=5):
        '''
        Computes slippage rates.
        flow_per_t      -  flow along that leg (list of projected flows per column (time))
        pos_membranes   -  position of membrane per kymo column (time)
        delta_t         -  distance between columns in kymos used to compute slippage
        '''
        slippage = np.zeros(len(pos_membranes)-delta_t)
        
        for t in range(len(pos_membranes)-delta_t): # for each column that allows us to jump delta_t forward
            t2 = t + delta_t
            delta_membrane = pos_membranes[t2] - pos_membranes[t]
            delta_flow = np.sum(flow_per_t[t:t2+1])
            slippage[t]=delta_flow-delta_membrane
        return slippage
            

    def plot_slippage(self, fig, delta_t=5, offset_from_membrane=0, length=None, move_window=False, smoothing_width=15):
        '''
        Plots the several statistics computed per column of the computed kymographs.
        fig                   -  the figure object to plot into
        delta_t               -  distance between columns in kymos used to compute slippage
        offset_from_membrane  -  stats to be computed in a stripe below membrane starting that many pixels below
        length                -  pixel height of stripe within which stats are computed (if None, stripe reaches until bottom)
        move_window           -  if true, the window we use to read flows is pushed around by the flow (static pos. otherwise)
        smoothing_width       -  width of sliding average window size
        '''
        fig.suptitle('Slippage', fontsize=16)
        
        slippages = []
        smoothed_slippages = []
        for legnum in range(self.num_legs):
            if legnum == 0:
                style = 'y-'
                style_reset = 'C*'
            else:
                style = 'c-'
                style_reset = 'y*'

            # read the segmented membrane location out of kymo_seg
            pos_membranes = np.zeros( len(self.kymographs[0][0]) )
            for col in range( len(self.kymographs[0][0]) ):
                pos_membranes[col] = np.argmax(self.kymo_seg[legnum][:,col])
            
            # get the average inward flow below the membrane
            #print np.shape(pos_membranes), pos_membranes
            minf, avgf, maxf, stdf, pos_intervals, resets = \
                        self.get_flow_stats(self.kymo_flows[legnum], pos_membranes+offset_from_membrane,length, move_window)
            
            # make kymo_seg transparent (in order to be able to plot on top of another kymo)
            kymo_seg_transp = np.ma.masked_where(self.kymo_seg[legnum] < .9, self.kymo_seg[legnum])
            
            # Finally we can compute the slippage!!!
            slippage_ys = self.get_slippage_rates(avgf, pos_membranes)
            slippage_xs = np.array(range(len(slippage_ys)))+int(delta_t/2)

            # - - - start the plotting - - - -
                                     
            ax = fig.add_subplot(2,self.num_legs,legnum+1)
            ax.imshow(self.kymo_myosin[legnum], plt.get_cmap('gray'))
            ax.imshow(kymo_seg_transp, plt.get_cmap('Reds'), vmin=0, vmax=255, alpha=.9)
            ax.plot(pos_intervals, style)
            if length is None:
                ax.plot(np.zeros_like(pos_membranes)+len(self.kymographs[0]), style)
            else:
                ax.plot(pos_intervals+length, style)
            if (len(resets)>0):
                ax.plot(zip(*resets)[0], zip(*resets)[1], style_reset, markersize=16)

            ax = fig.add_subplot(2,self.num_legs,self.num_legs+legnum+1)
            ax.set_ylim([-25,25])
            ax.plot(np.zeros_like(avgf), color='gray')
            ax.plot(slippage_xs, slippage_ys, color='blue')
            # flupp
            running_average_ys = self.moving_average(slippage_ys, n=smoothing_width)
            running_average_xs = np.array(range(len(running_average_ys))) + int((delta_t+smoothing_width)/2)
            ax.plot(running_average_xs, running_average_ys, color='red')
            
            # store plotted values for later use
            slippages.append( (slippage_xs, slippage_ys) )
            smoothed_slippages.append( (running_average_xs, running_average_ys) )
        return (slippages, smoothed_slippages)

