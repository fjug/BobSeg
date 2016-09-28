import numpy as np
import bresenham as bham
import maxflow
import math
from tifffile import imread, imsave
import cPickle as pickle

from netsurface2d import NetSurf2d

import matplotlib as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

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

        better_centers = [None] * len(self.images)
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

    # *****************************************************************************************************
    # *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD ***
    # *****************************************************************************************************
    
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
            
    # **************************************************************************************************
    # *** VISUALIZATIONS *** VISUALIZATIONS *** VISUALIZATIONS *** VISUALIZATIONS *** VISUALIZATIONS ***
    # **************************************************************************************************
    
    def plot_minmax( self, frame, ax ):
        ax.imshow(self.images[frame])
        
        for oid in range(len(self.object_names)):
            patches = [] # collects patches to be plotted

            center = self.netsurfs[oid][frame].center
            #min_radius = self.netsurfs[oid][frame].min_radius
            #max_radius = self.netsurfs[oid][frame].max_radius
            min_radius = self.object_min_surf_dist[oid][frame]
            max_radius = self.object_max_surf_dist[oid][frame]
            
            ax.scatter(center[0],center[1], c='y', marker='o')
            patches.append( Ellipse((center[0],center[1]),
                                    width=min_radius[0],
                                    height=min_radius[1]) )
            patches.append( Ellipse((center[0],center[1]),
                                    width=max_radius[0],
                                    height=max_radius[1]) )
            p = PatchCollection(patches, cmap=plt.cm.jet, alpha=0.4, color='green')
            ax.add_collection(p)

    def plot_result( self, frame, ax ):
        ax.imshow(self.images[frame])
        
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
            p = PatchCollection(patches, cmap=plt.cm.jet, alpha=0.4, color='green')
            ax.add_collection(p)
