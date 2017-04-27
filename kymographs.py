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
    
    num_legs = 4         # the number of legs of the spider
    ea_ep_leg_index = -1 # index of the leg between Ea and Ep cell (-1 means unknown)
    center = (100,100)   # the center point of the current spider
    length = 100         # the length of each spider leg
    
    kymographs = [None]*num_legs   # the computed Kymographs for the membrane channel
    kymo_myosin = [None]*num_legs  # the computed Kymographs for the myosin channel
    kymo_seg = [None]*num_legs     # the computed Kymographs for the segmentation channel
    kymo_flows = [None]*num_legs   # the computed flow Kymographs for the flow
    
    membrane = [None]*num_legs   # the position of the membrane for each spider leg
    fiducials = [None]*num_legs  # the position of fiducials along spider legs
    

    def __init__( self, num_legs=8 ):
        self.num_legs = num_legs
        
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

    def move_fiducial(self, flow_kymo, initial_pos):
        '''
        Moves a fiducial dot in a flow kymograph.
        flow_kymo    - a flow Kymograph (x-axis is time, y-axis is space, intensities are y-projected flow)
        initial_pos  - an integer defining the y-position of the fiducial dot in the flow Kymographs first column
        '''
        pos = initial_pos
        positions = [pos]
        for col in range(1,flow_kymo.shape[1]): # 1 because flow at t==0 is blank
            pos += flow_kymo[int(round(pos)),col]
            pos = min(len(flow_kymo)-1,pos)
            positions.append(pos)
        return positions
    
    def set_ea_ep_leg(self, index):
        '''
        Sets the index of the lag that points toward the Ea/Ep cell boundary (-1 means undefined)
        '''
        self.ea_ep_leg_index = index
    
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
        self.length = length
        
    def get_leg_length(self):
        '''
        Returns the set leg length.
        '''
        return self.length
    
    def get_leg_line(self, legnum):
        '''
        Returns a list if two (x,y)-coordinate tuples defining the start and endpoint of the desired spiderleg.
        '''
        dx = int( math.sin(math.pi*2*legnum/self.num_legs)*self.length )
        dy = int( math.cos(math.pi*2*legnum/self.num_legs)*self.length )
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
        line_pixels = bham.bresenhamline(np.array([p_start]),np.array([p_end]))
                               
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
            if legnum == self.ea_ep_leg_index:
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
        
        ax = fig.add_subplot(2,self.num_legs+1,1)
        ax.imshow(frame_first)
        self.plot_spider_on_axis(ax)
        ax = fig.add_subplot(2,self.num_legs+1,self.num_legs+1+1)
        ax.imshow(frame_last)
        self.plot_spider_on_axis(ax)

        for legnum in range(self.num_legs):
            if legnum == self.ea_ep_leg_index:
                style = 'y.'
            else:
                style = 'c.'
                
            pos = pos_fiducial
            if rel_to_membrane:
                pos+=np.argmax(self.kymo_seg[legnum][:,0])

            ax = fig.add_subplot(2,self.num_legs+1,legnum+2)
            ax.imshow(self.kymographs[legnum], plt.get_cmap('gray'))
            kymo_seg_transp = np.ma.masked_where(self.kymo_seg[legnum] < .9, self.kymo_seg[legnum])
            ax.imshow(kymo_seg_transp, plt.get_cmap('Reds'), vmin=0, vmax=1.5, alpha=.9)
            ax.plot(self.move_fiducial(self.kymo_flows[legnum],pos), style)
            #ax.axis('off')
            
            ax = fig.add_subplot(2,self.num_legs+1,self.num_legs+1+legnum+2)
            #ax.imshow(self.kymo_flows[legnum], plt.get_cmap('gray'))
            ax.imshow(self.kymo_myosin[legnum], plt.get_cmap('gray'))
            ax.imshow(kymo_seg_transp, plt.get_cmap('Reds'), vmin=0, vmax=1.5, alpha=.9)
            ax.plot(self.move_fiducial(self.kymo_flows[legnum],pos), style)
            #ax.axis('off')
