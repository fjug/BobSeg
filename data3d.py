import numpy as np
import bresenham as bham
import maxflow
import math
from tifffile import imread, imsave
import cPickle as pickle
import cv2

from netsurface2d import NetSurf2d
from netsurface2dt import NetSurf2dt

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
import pylab

class Data3d:
    """
    Implements a container to hold 2d+t (3d) time-lapse datasets.
    Time points in such datasets can conveniently be segmented via NetSurface2d.
    """
    
    silent = True
    images = []
    pixelsize=(1.,1.,1.)
    object_names = []
    object_seedpoints = {}
    object_areas = {}
    object_min_surf_dist = {}
    object_max_surf_dist = {}

    netsurfs = {}
    netsurf2dt = {} # instances of NetSurf2dt
    
    # global segmentation parameters (for NetSurf2d)
    # (set from outside using method 'set_seg_params')
    num_columns = 60
    K = 30
    max_delta_k = 4
    
    colors_grey = [(1.-.15*i,1.-.15*i,1.-.15*i) for i in range(6)]
    colors_red = [(1.,.2,.2*i) for i in range(6)]
    colors_gold = [(1.,.8,.15*i) for i in range(6)]
    colors_yellow = [(1.,1.,.9-.15*i) for i in range(6)]
    colors_green = [(.45,1.,.4+.1*i) for i in range(6)]
    colors_blue = [(.4,1.,1.3+1*i) for i in range(6)]
    colors_darkblue = [(.1,.3,1.0-.1*i) for i in range(6)]
    colors_diverse = [ colors_green[0], colors_red[0], colors_blue[0], colors_gold[0], colors_yellow[0],colors_grey[1] ]
    
    def __init__( self, segimages, pixelsize=None, silent=True ):
        """
        Parameters:
            segimages   -  a 2d+t (3d) image stack (t,y,x)
            pixelsize   -  pixel calibration (e.g. for area computation)
            silent      -  if True, no (debug/info) outputs will be printed on stdout
        """
        self.silent = silent
        self.images = [None] * segimages.shape[0]
        for i in range (segimages.shape[0]):
            self.images[i] = segimages[i]
        if not pixelsize is None: self.pixelsize = pixelsize

    # ***********************************************************************************************
    # *** SEGMENTATION STUFF *** SEGMENTATION STUFF *** SEGMENTATION STUFF *** SEGMENTATION STUFF ***
    # ***********************************************************************************************

    def set_seg_params( self, num_columns, K, max_delta_k ):
        self.num_columns = num_columns
        self.K = K
        self.max_delta_k = max_delta_k
        
    def init_object( self, name ):
        """
        Adds an (empty) object definition to this dataset.
        Returns the id of the added object.
        """
        oid = self.get_object_id( name )
        if oid == -1: # not found
            oid = len(self.object_names)
            self.object_names.append(name)
        self.object_seedpoints[oid] = [None] * len(self.images)
        self.object_min_surf_dist[oid] = [(0,0)] * len(self.images)
        self.object_max_surf_dist[oid] = [(100,100)] * len(self.images)
        self.object_areas[oid] = [0] * len(self.images)
        self.netsurfs[oid] = [None] * len(self.images)
        self.netsurf2dt[oid] = None
        return oid
    
    def get_object_id( self, name ):
        for i,n in enumerate(self.object_names):
            if name == n:
                return i
        return -1
        
    def add_object_at ( self, oid, min_rs, max_rs, frame, seed, frame_to=None, seed_to=None, segment_it=False ):
        """
        Makes a given (already added) object exist at a frame (or a sequence of consecutive frames).
        Parameters:
            oid         -  the object id as returned by 'add_object'
            min_rs      -  three-tuple defining min # of pixels from seed to look for the object surface
            max_rs      -  three-tuple defining max # of pixels from seed to look for the object surface
            frame       -  the frame index at which the object occurs (is visible)
            seed        -  three-tuple defining (x,y,z) coordinates used to seed the segmentation of this object
            frame_to    -  if given and >frame, all frames in [frame,frame_end] will me marked to contain this object
            seed_to     -  if frame_end is given 'seed_end' defines the center point at frame_end plus all intermediate
                           time points will linearly interpolate between 'seed' and 'seed_end'. If not given, 'seed' is 
                           used for all time points in [frame,frame_end]
        """
        if frame_to is None: 
            frame_to = frame
        if seed_to is None:
            seed_to = seed
            
        assert frame >= 0
        assert frame < len(self.images)
        assert frame_to >= 0
        assert frame_to < len(self.images)
        assert frame <= frame_to
        
        for i in range(frame,frame_to+1):
            self.object_min_surf_dist[oid][i] = min_rs
            self.object_max_surf_dist[oid][i] = max_rs        
            self.object_seedpoints[oid][i] = self.interpolate_points(np.array(seed), 
                                                                     np.array(seed_to), 
                                                                     float(i-frame)/(1+frame_to-frame))
            if not self.silent:
                print 'Added appearance for "'+str(self.object_names[oid])+ \
                      '" in frame', i, \
                      'with seed coordinates', self.object_seedpoints[oid][i]
            if segment_it: self.segment_frame( oid, i )

    def interpolate_points( self, start, end, fraction ):
        return np.round( start + (end-start)*fraction )
    
    def segment_frame( self, oid, f ):
        """
        Segments object oid in frame f.
        """
        assert oid>=0
        assert oid<len(self.object_names)
        assert f>=0
        assert f<len(self.images)
        
        try:
            self.netsurfs[oid][f] = None
        except:
            print 'LAZY INIT NETSURFS'
            self.netsurfs[oid] = [None] * len(self.images)
        
        self.netsurfs[oid][f] = NetSurf2d(self.num_columns, K=self.K, max_delta_k=self.max_delta_k)
        optimum = self.netsurfs[oid][f].apply_to(self.images[f], 
                                                 self.object_seedpoints[oid][f], 
                                                 self.object_max_surf_dist[oid][f], 
                                                 min_radius=self.object_min_surf_dist[oid][f])
        self.object_areas[oid][f] = self.netsurfs[oid][f].get_area( self.pixelsize )
        if not self.silent:
            print '      Optimum energy: ', optimum
            ins, outs = self.netsurfs[oid][f].get_counts()
            print '      Nodes in/out: ', ins, outs
            print '      Area: ', self.object_areas[oid][f]
            
    def segment2dt( self, oid, max_radial_delta=2 ):
        '''
        Segments entire time series using NetSurf2dt (increases temporal consistency)
        Note: changes self.object_areas!
        '''
        self.netsurf2dt[oid] = NetSurf2dt(self.num_columns, 
                                          K=self.K, 
                                          max_delta_k_xy=self.max_delta_k, 
                                          max_delta_k_t=max_radial_delta)
        optimum = self.netsurf2dt[oid].apply_to(self.images, 
                                           self.object_seedpoints[oid], 
                                           self.object_max_surf_dist[oid][0], # note: frame 0 currently rules them all
                                           min_radius=self.object_min_surf_dist[oid][0])
        for t in range(len(self.images)):
            self.object_areas[oid][t] = self.netsurf2dt[oid].get_area( t, self.pixelsize )
            if not self.silent:
                print 'Results for frame %d:'%(t)
                print '      Optimum energy: ', optimum
                print '      Area: ', self.object_areas[oid][t]
        
            
    # ***************************************************************************************************
    # *** TRACKING&REFINEMENT *** TRACKING&REFINEMENT *** TRACKING&REFINEMENT *** TRACKING&REFINEMENT ***
    # ***************************************************************************************************
    
    def get_center_estimates( self, oid, frames=None, set_as_new=False ):
        """
        Computes a better center point then the one used for segmenting object 'oid'.
        If 'set_as_new==True', these new center points will be set an new seed points.
        """
        assert oid>=0
        assert oid<len(self.object_names)
        
        if frames is None:
            frames = range(len(self.images)) 

        better_centers = self.object_seedpoints[oid]
        for f in frames:
            if not self.object_seedpoints[oid][f] is None:
                assert not self.netsurfs[oid][f] is None # segmentation must have been performed
                netsurf = self.netsurfs[oid][f]
                better_centers[f] = np.array(netsurf.get_surface_point(0))
                for i in range(1,netsurf.num_columns):
                    better_centers[f] += netsurf.get_surface_point(i)
                better_centers[f] /= netsurf.num_columns
                if not self.silent:
                    print '    Updated center to',better_centers[f]
        # update seedpoints if that was desired
        if set_as_new: self.object_seedpoints[oid] = better_centers
        return better_centers
    
    def track( self, oid, seed_frame, target_frames, recenter_iterations=1 ):
        """
        For the object with the given id this function tries to fill in missing frames.
        Note: this will only work if the object seed in seed_frame lies within the desired object in
        the first target_frames and the re-evaluated center in each consecutive target frame (iterated).
        Parameters:
            oid           -  object id that should be tracked
            seed_frame    -  frame id that was previously seeded (using add_object_at)
            target_frame  -  list of frame ids the object should be tracked at
            recenter_iterations  -  how many times should the new center be looked for iteratively?
        """
        assert oid>=0
        assert oid<len(self.object_names)

        seed = self.object_seedpoints[oid][seed_frame]
        min_rs = self.object_min_surf_dist[oid][seed_frame]
        max_rs = self.object_max_surf_dist[oid][seed_frame]
        for f in target_frames:
            self.add_object_at( oid, min_rs, max_rs, f, seed )
            self.segment_frame( oid, f )
            for i in range(recenter_iterations):
                self.get_center_estimates( oid, [f], set_as_new=True )
            seed = self.object_seedpoints[oid][f]

    # ********************************************************************************************
    # *** FLOW ***  FLOW ***  FLOW ***  FLOW ***  FLOW ***  FLOW ***  FLOW ***  FLOW ***  FLOW *** 
    # ********************************************************************************************
    
    def compute_flow( self, flowchannel, segchannel, folder=None, show=False, inline=False ):
        '''
        Computes and optionally displays flow and segmentation.
        Parameters:
            flowchannel   -  the images (t,y,x) the flow should be computed on
            segchannel    -  the images (t,y,x) used for segmentation (might coincide with self.images
                             but are not a list of 2d images but an 3d image stack.
                             Note: segchannel is here ONLY used for visualization purposes!!!
            folder        -  If not 'None' this folder will be used to store all visualized frames.
            show          -  Rendered frames will be shown on screen or inline if set to 'True'
            inline        -  False: openCV window will be used to display rendering
                             True: matplotlib is used (can be used in jupyter inline mode!)
        '''
        assert flowchannel.shape[0] == len(self.images)
        assert flowchannel.shape[0] == segchannel.shape[0]
        
        show_inline = inline
        nihilation_radius = 15 # how big is the center area that kills fiducials
        do_respawn = False     # should killed fiducials respawn?
        respawn_margin = 20    # how many pixels from the cell outline should a killed fiucial respawn?
        
        self.flows = [None]*len(self.images)
        
        if show_inline:
            from IPython.display import clear_output
            pylab.rcParams['figure.figsize'] = (25, 10)
            fig = plt.figure()

        prvs = flowchannel[0]
        hsv_shape = (prvs.shape[0],prvs.shape[1],3)
        hsv = np.zeros(hsv_shape)
        hsv[...,1] = 255

        # collect the fiducial dots
        dots = []
        for oid in range(0,len(self.object_names)):
            dots.extend( self.get_radialdots_in(0,oid,2,respawn_margin) ) # self.get_griddots_in(0,oid,spacing=15) )
            
        dot_history = [dots]
        for f in range(flowchannel.shape[0]):
            nxt = flowchannel[f]

            flow = cv2.calcOpticalFlowFarneback(prev=prvs,
                                                next=nxt,
                                                pyr_scale=0.5,
                                                levels=3,
                                                winsize=5,
                                                iterations=15,
                                                poly_n=5,
                                                poly_sigma=1.5,
                                                flags=1)
            self.flows[f]=flow

            cells = []
            for oid in range(len(self.object_names)):
                if self.netsurf2dt is None:
                    cells.append( self.get_result_polygone(oid,f) )
                else:
                    cells.append( self.get_result_polygone_2dt(oid,f) )

            centers = []
            for oid in range(0, len(self.object_names)):
                centers.append( self.object_seedpoints[oid][f])

            outframe, dots = self.draw_flow(flowchannel[f],
                                       segchannel[f],
                                       flow,
                                       dots=dots,
                                       dothist=dot_history,
                                       polygones=cells,
                                       show_flow_vectors=False,
                                       centers=centers,
                                       nihilation_radius=nihilation_radius)
            dot_history.insert(0,dots)

            # TESTCODE - DOT REPLACEMENT
            potentialnewdots = []
            for oid in range(0, len(self.object_names)):
                potentialnewdots.extend( self.get_radialdots_in(f, oid, 2, respawn_margin) )
            for dot_index, dot in enumerate(dots):
                for oid in range(0, len(self.object_names)):
                    if np.linalg.norm( self.object_seedpoints[oid][f] - dot ) < nihilation_radius:
                        if do_respawn:
                            dots[dot_index] = potentialnewdots[dot_index]
                        else:
                            dots[dot_index] = (-1,-1)
                        for dh in dot_history:
                            if do_respawn:
                                dh[dot_index] = potentialnewdots[dot_index]
                            else:
                                dh[dot_index] = (-1,-1)


            rgbframe = cv2.cvtColor(outframe, cv2.COLOR_BGR2RGB)

            # save frames if desired
            if not folder is None:
                cv2.imwrite(folder+'frame%04d.png'%(f), outframe)

            if show_inline:
                pylab.axis('off')
                pylab.title("flow")
                pylab.imshow(rgbframe)
                pylab.show()
                clear_output(wait=True)
                # optional quick exit (DEBUGGING)
                if False and f==3:
                    break
            else:
                cv2.imshow('flow',outframe)
                k = cv2.waitKey(25) & 0xff
                if k == 27: # ESC
                    break

            prvs = nxt

        if not show_inline:
            cv2.destroyAllWindows()
    
    # ***************************************************************************************
    # *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD ***
    # ***************************************************************************************
    
    def save( self, filename ):
        dictDataStorage = {
            'silent'               : self.silent,
            'object_names'         : self.object_names,
            'object_seedpoints'    : self.object_seedpoints,
            'object_areas'         : self.object_areas,
            'object_min_surf_dist' : self.object_min_surf_dist,
            'object_max_surf_dist' : self.object_max_surf_dist,
            'K'                    : self.K,
            'max_delta_k'          : self.max_delta_k,
        }
        with open(filename,'w') as f:
            pickle.dump(dictDataStorage,f)

    def load( self, filename, compute_netsurfs=True ):
        with open(filename,'r') as f:
            dictDataStorage = pickle.load(f)

        self.silent = dictDataStorage['silent']
        self.object_names = dictDataStorage['object_names']
        self.object_seedpoints = dictDataStorage['object_seedpoints']
        self.object_areas = dictDataStorage['object_areas']
        self.object_min_surf_dist = dictDataStorage['object_min_surf_dist']
        self.object_max_surf_dist = dictDataStorage['object_max_surf_dist']
        self.K = dictDataStorage['K']
        self.max_delta_k = dictDataStorage['max_delta_k']
        if compute_netsurfs: self.segment()
         
    # *********************************************************************************************
    # *** NUMBER_CRUNCH *** NUMBER_CRUNCH *** NUMBER_CRUNCH *** NUMBER_CRUNCH *** NUMBER_CRUNCH *** 
    # *********************************************************************************************
    def get_result_polygone( self, oid, frame ):
        points=[]
        col_vectors = self.netsurfs[oid][frame].col_vectors
        netsurf = self.netsurfs[oid][frame]
        for i in range( len(col_vectors) ):
            points.append( netsurf.get_surface_point(i) )
        return points
    
    def get_result_polygone_2dt( self, oid, frame ):
        points=[]
        col_vectors = self.netsurf2dt[oid].col_vectors
        netsurf2dt = self.netsurf2dt[oid]
        for i in range( len(col_vectors) ):
            points.append( netsurf2dt.get_surface_point(frame,i) )
        return points
    
    def get_k_over_time( self, oid ):
        k_over_time = np.zeros( (self.num_columns, len(self.images)) )
        for f in range( len(self.images) ):
            for i in range( self.num_columns ):
                k_over_time[i,f] = self.netsurf2dt[oid].get_surface_index(f,i)
        return k_over_time
    
    def get_dist_to_center( self, oid, frame ):
        c = self.object_seedpoints[oid][frame]
        poly = self.get_result_polygone_2dt( oid, frame )
        dists = []
        for p in poly:
            dists.append( ((p[0]-c[0])**2+(p[1]-c[1])**2)**.5 )
        return np.array(dists)
    
    def get_radial_velocities( self, oid, frame ):
        dist_t   = self.get_dist_to_center( oid, frame )
        dist_tp1 = self.get_dist_to_center( oid, frame+1 )
        return dist_tp1-dist_t
    
    def get_column_flowvectors( self, oid, frame, column_id ):
        '''
        Returns all flowvector for one column of the net surface graph.
        The length of the returned list of 2-tupel (flow-vectors x,y) corresponds to the 
        number of pixels along the bresenham line from this objects seedpoint and the point
        the net surface flow found the outline.
        '''
        flow = self.flows[frame]
        point_c = self.object_seedpoints[oid][frame]
        point_b = self.netsurfs[oid][frame].get_surface_point(column_id)
        coords = bham.bresenhamline(np.array([point_c]), np.array([point_b]))
        vectors = []
        for c in coords:
            fx,fy = flow[c[1],c[0]].T
            vectors.append( (fx,fy) )
        return vectors
        
    def get_all_flowvectors( self, oid, frame ):
        '''
        Returns a vector the length of 'self.num_columns'.
        Each entry contains all flowvector for one column of the net surface graph.
        The length of the returned list of 2-tupel (flow-vectors x,y) corresponds to the 
        number of pixels along the bresenham line from this objects seedpoint and the point
        the net surface flow found the outline.
        '''
        colvectors = []
        for i in range(self.num_columns):
            colvectors.append( self.get_column_flowvectors( oid, frame, i ) )
        return colvectors
        
    # **************************************************************************************************
    # *** VISUALIZATIONS *** VISUALIZATIONS *** VISUALIZATIONS *** VISUALIZATIONS *** VISUALIZATIONS ***
    # **************************************************************************************************
    
    def plot_minmax( self, frame, ax ):
        ax.imshow(self.images[frame], plt.get_cmap('gray'))
        
        for oid in range(len(self.object_names)):
            patches = [] # collects patches to be plotted

            center = self.netsurfs[oid][frame].center
            min_radius = self.object_min_surf_dist[oid][frame]
            max_radius = self.object_max_surf_dist[oid][frame]
            
            # ax.scatter(center[0],center[1], c='y', marker='x')
            patches.append( Ellipse((center[0],center[1]),
                                    width=(min_radius[0]*2),
                                    height=(min_radius[1])*2) )
            patches.append( Ellipse((center[0],center[1]),
                                    width=(max_radius[0]*2),
                                    height=(max_radius[1]*2)) )
            p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.3, color='yellow')
            ax.add_collection(p)

    def plot_result( self, frame, ax ):
        ax.imshow(self.images[frame], plt.get_cmap('gray'))
        
        for oid in range(len(self.object_names)):
            patches = [] # collects patches to be plotted
            surface=[] # will collect the highest v per column in here
            
            col_vectors = self.netsurfs[oid][frame].col_vectors
            center = self.netsurfs[oid][frame].center
            min_radius = self.netsurfs[oid][frame].min_radius
            max_radius = self.netsurfs[oid][frame].max_radius
            netsurf = self.netsurfs[oid][frame]
            for i in range( len(col_vectors) ):
                surface.append((0,0))
                surface[i] = netsurf.get_surface_point(i)
            polygon = Polygon(surface, True)
            patches.append(polygon)
            p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4, color='green')
            ax.add_collection(p)

    def plot_2dt_result( self, frame, ax ):
        ax.imshow(self.images[frame], plt.get_cmap('gray'))
        
        for oid in range(len(self.object_names)):
            patches = [] # collects patches to be plotted
            surface=[] # will collect the highest v per column in here
            
            col_vectors = self.netsurf2dt[oid].col_vectors
            center = self.netsurf2dt[oid].centers[frame]
            min_radius = self.netsurf2dt[oid].min_radius
            max_radius = self.netsurf2dt[oid].max_radius
            netsurf2dt = self.netsurf2dt[oid]
            for i in range( len(col_vectors) ):
                surface.append((0,0))
                surface[i] = netsurf2dt.get_surface_point(frame,i)
            polygon = Polygon(surface, True)
            patches.append(polygon)
            p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4, color='green')
            ax.add_collection(p)

    def get_radialdots_in( self, frame, oid, spacing=5, pixels_inwards=5 ):
        '''
        Returns list of (x,y) coordinates placed at given distance inside a segmented polygon.
        Note that these points will only be placed if they where not shooting over the center.
            frame            - guess
            oid              - object id
            spacing=5        - how many col-vectors to jump (5 === 20% of radial dots will be created)
            pixels_inwards=5 - how many pixels parallel (and inwards) from polygon will markers be placed?
        '''
        points=[]
        
        netsurf = self.netsurfs[oid][frame]
        polypoints = np.array( self.get_result_polygone(oid,frame) )
        
        cx = netsurf.center[0]
        cy = netsurf.center[1]
        for i in range( 0, len(netsurf.col_vectors), spacing ):
            x = polypoints[i,0]
            y = polypoints[i,1]
            dx = netsurf.col_vectors[i][0]
            dy = netsurf.col_vectors[i][1]
            x_new = int(x - dx * pixels_inwards)
            y_new = int(y - dy * pixels_inwards)
            # only if not shooting over center, add point!
            if np.sign(x-cx) - np.sign(x_new-cx) == 0:
                points.append((x_new, y_new))
                
        return points
    
    def get_griddots_in( self, frame, oid, spacing=5 ):
        points=[]
        
        polypoints = np.array( self.get_result_polygone(oid,frame) )
        poly = Path( polypoints, closed=True )
        
        minx = np.min(polypoints[:,0])
        maxx = np.max(polypoints[:,0])
        miny = np.min(polypoints[:,1])
        maxy = np.max(polypoints[:,1])
        for x in range(minx,maxx,spacing):
            for y in range(miny,maxy,spacing):
                if poly.contains_point((x,y)):
                    points.append((x,y))
                    
        return points
    
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
        h,w = im.shape[:2]
        y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
        fx,fy = flow[y,x].T

        # create image and draw
        vis = cv2.cvtColor(np.uint8(np.zeros_like(im)),cv2.COLOR_GRAY2BGR)
        im = im-np.min(im)
        im2 = im2-np.min(im2)
        vis[:,:,1] = (255*im)/np.max(im)
        vis[:,:,2] = (255*im2)/np.max(im2)

        # flow arrows
        if ( show_flow_vectors ):
            lines = vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
            lines = int32(lines)

            for (x1,y1),(x2,y2) in lines:
                cv2.line(vis,(x1,y1),(x2,y2),(0,150,150),1)
                cv2.circle(vis,(x1,y1),1,(0,150,150), -1)

        # draw polygones
        for polygone in polygones:
            cv2.polylines(vis, np.array([polygone], 'int32'), 1, (208,224,64), 2)

        # draw fiducial nihilation zone
        for (x,y) in centers:
            cv2.circle(vis, (int(x),int(y)), int(nihilation_radius), (208,224,128), 2)

        newdots = []
        for dot_idx, dot in enumerate(dots):
            # only show the dots that are 'alive' (dead ones are (-1,-1))
            if dot[0] == -1:
                continue

            # compute new dot
            try:
                fx,fy = flow[int(dot[1]),int(dot[0])].T
            except:
                fx = 0
                fy = 0
            newx = max(0,dot[0]+fx)
            newy = max(0,dot[1]+fy)
            newx = min(im.shape[1]-1,newx)
            newy = min(im.shape[0]-1,newy)
            newdot = ( newx, newy )

            # history lines
            color = (255,128,128)
            intdot = ( int(newdot[0]), int(newdot[1]) )
            p1 = intdot
            for time_idx,histdots in enumerate(dothist):
                hdot = histdots[dot_idx]
                p2 = ( int(hdot[0]), int(hdot[1]) )
                if p1[0] != -1 and p2[0] != -1: # don't show trail to 'dead' fiducials
                    cv2.line(vis, p1, p2, color, 1)
                color = tuple(np.array(color)-[10,10,10])
                if min(color) < 0: 
                    break
                p1 = p2

            # point (after to draw on top of history lines etc)
            cv2.circle(vis,intdot, 3, (0,165,255), 1)

            newdots.append(newdot)

        return vis, np.array(newdots)

    def draw_segmentation(self, im, show_centers=True, dont_use_2dt=False, folder=None, inline=False):
        '''
        Draws movie frames for segmented objects with/without center points being visible.
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
            vis = cv2.cvtColor(np.zeros_like(im[f]),cv2.COLOR_GRAY2BGR)
            vis[:,:,0] = 255.0*im[f]/np.max(im[f])
            vis[:,:,1] = 255.0*im[f]/np.max(im[f])
            vis[:,:,2] = 255.0*im[f]/np.max(im[f])
            
            # show center dot
            if show_centers:
                for oid in range(len(self.object_names)):
                    color = int(128+128./len(self.object_names)*(oid+1))
                    for f2 in range(f+1):
                        center = tuple(self.object_seedpoints[oid][f2])
                        cv2.circle(vis, center, 3, (0,color,0), 1)
                        centers.append(center)
            
                    # show best fitting circle
                    r = 0.
                    for i in range( self.num_columns ):
                        r += self.netsurf2dt[oid].get_surface_index(f,i)
                    r /= self.num_columns
                    r /= self.K
                    r *= self.object_max_surf_dist[oid][f][0]-self.object_min_surf_dist[oid][f][0]
                    r += self.object_min_surf_dist[oid][f][0]
                    radii.append(r)

                    cv2.circle(vis,tuple(self.object_seedpoints[oid][f2]), int(r), (0,color,0), 1)
            
            # retrieve polygones
            polygones = []
            for oid in range(len(self.object_names)):
                if self.netsurf2dt is None or dont_use_2dt:
                    polygones.append( self.get_result_polygone(oid,f) )
                else:
                    polygones.append( self.get_result_polygone_2dt(oid,f) )
            all_polygones.append(polygones)
                    
            # draw polygones
            for polygone in polygones:
                cv2.polylines(vis, np.array([polygone], 'int32'), 1, (255,0,0), 2)

            rgbframe = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            frames.append(rgbframe)

            # save frames if desired
            if not folder is None:
                cv2.imwrite(folder+'frame%4d.png'%(f), vis)

            if inline:
                pylab.axis('off')
                pylab.title("segmentation")
                pylab.imshow(rgbframe)
                pylab.show()
                clear_output(wait=True)
                # optional quick exit (DEBUGGING)
                if False and f==3:
                    break
            else:
                cv2.imshow('segmentation',vis)
                k = cv2.waitKey(25) & 0xff
                if k == 27: # ESC
                    break

        if not inline:
            cv2.destroyAllWindows()
            
        return frames, centers, all_polygones, radii
    
    def create_segmentation_image(self, dont_use_2dt=False):
        segimgs = np.zeros_like(self.images)
        for f in range(len(self.images)):
            vis = np.zeros((np.shape(segimgs)[1],np.shape(segimgs)[2],3), np.uint8)

            # retrieve polygones
            polygones = []
            for oid in range(len(self.object_names)):
                if self.netsurf2dt is None or dont_use_2dt:
                    polygones.append( self.get_result_polygone(oid,f) )
                else:
                    polygones.append( self.get_result_polygone_2dt(oid,f) )

            # draw polygones
            for polygone in polygones:
                cv2.polylines(vis, np.array([polygone], 'int32'), 1, (128,128,128), 2)
                cv2.polylines(vis, np.array([polygone], 'int32'), 1, (255,255,255), 1)


            segimgs[f] = vis[:,:,0]
        return segimgs