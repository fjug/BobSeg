import cPickle as pickle

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon

import bresenham as bham
from netsurface2d import NetSurf2d
from netsurface2dt import NetSurf2dt


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
    
    def compute_flow( self, flowchannel ):
        assert flowchannel.shape[0] == len(self.images)

        self.flows = [None] * len(self.images)
        prvs = flowchannel[0]
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
            self.flows[f] = flow
            prvs = nxt
            print '.',
        print ' ...done!'
        return self.flows

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