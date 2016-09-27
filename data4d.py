import numpy as np
import bresenham as bham
import maxflow
import math
from tifffile import imread, imsave
import cPickle as pickle

from spimagine import volfig, volshow
#from spimagine import EllipsoidMesh, Mesh

from netsurface3d import NetSurf3d

class Data4d:
    """
    Implements a container to hold 3d+t (4d) time-lapse datasets.
    Time points in such datasets can conveniently be segmented via NetSurface3d and 
    visualized in Spimagine.
    """
    
    silent = True
    images = [None]
    object_names = []
    object_visibility = {}
    object_seedpoints = {}
    object_volumes = {}
    object_min_surf_dist = {}
    object_max_surf_dist = {}

    netsurfs = {}
    spimagine = None
    current_fame = None
    
    # global segmentation parameters (for NetSurf3d)
    # (set from outside using method 'set_seg_params')
    K = 30
    max_delta_k = 3
    
    colors_grey = [(1.-.15*i,1.-.15*i,1.-.15*i) for i in range(6)]
    colors_red = [(1.,.2,.2*i) for i in range(6)]
    colors_gold = [(1.,.8,.15*i) for i in range(6)]
    colors_yellow = [(1.,1.,.9-.15*i) for i in range(6)]
    colors_green = [(.45,1.,.4+.1*i) for i in range(6)]
    colors_blue = [(.4,1.,1.3+1*i) for i in range(6)]
    colors_darkblue = [(.1,.3,1.0-.1*i) for i in range(6)]
    colors_diverse = [ colors_green[0], colors_red[0], colors_blue[0], colors_gold[0], colors_yellow[0],colors_grey[1] ]
    
    def __init__( self, filenames, silent=True ):
        """
        Parameters:
            filenames   -  list of filenames (one per time point)
            silent      -  if True, no (debug/info) outputs will be printed on stdout
        """
        self.silent = silent

        # load pickeled unit sphere sampling
        with open('sphere_sampling.pkl','r') as f:
            dictSphereData = pickle.load(f)

        # sampling parameters
        self.vectors = dictSphereData['points']
        self.neighbors = dictSphereData['neighbors']
        self.neighbors_of = dictSphereData['neighbors_of']
        self.triangles = dictSphereData['triangles']
        
        # load images
        self.load_from_files( filenames )

    def set_seg_params( self, K, max_delta_k ):
        self.K = K
        self.max_delta_k = max_delta_k
        
    def load_from_files( self, filenames ):
        self.images = [None]*len(filenames)
        for i in range(len(filenames)):
            self.images[i] = imread(filenames[i])
            if not self.silent:
                print 'Dimensions of frame', i, ': ', self.images[i].shape
            
    def init_object( self, name ):
        """
        Adds an (empty) object definition to this dataset.
        Returns the id of the added object.
        """
        oid = len(self.object_names)
        self.object_names.append(name)
        self.object_visibility[oid] = [False] * len(self.images)
        self.object_seedpoints[oid] = [None] * len(self.images)
        self.object_volumes[oid] = [0] * len(self.images)
        self.netsurfs[oid] = [None] * len(self.images)
        return oid
    
    def get_object_id( self, name ):
        for i,n in enumerate(self.object_names):
            if name == n:
                return i
        
    def add_object_at ( self, oid, min_rs, max_rs, frame, seed, frame_to=None, seed_to=None ):
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
        
        self.object_min_surf_dist[oid] = min_rs
        self.object_max_surf_dist[oid] = max_rs
        
        for i in range(frame,frame_to+1):
            self.object_visibility[oid][i] = True
            self.object_seedpoints[oid][i] = self.interpolate_points(np.array(seed), 
                                                                     np.array(seed_to), 
                                                                     float(i-frame)/(1+frame_to-frame))
            self.netsurfs[oid][i] = NetSurf3d(self.vectors, self.triangles, self.neighbors_of, 
                                              K=self.K, max_delta_k=self.max_delta_k)
            if not self.silent:
                print 'Added appearance for "'+str(self.object_names[oid])+ \
                      '" in frame', i, \
                      'with seed coordinates', self.object_seedpoints[oid][i]

    def interpolate_points( self, start, end, fraction ):
        return np.round( start + (end-start)*fraction )
    
    def segment( self, oids=None, pixelsize=(1.,1.,1.) ):
        """
        Segments all added object occurances using NetSurf3d.
        If oids is not given (None) this will be done for ALL objects at ALL time points.
        """
        if oids is None:
            oids = range(len(self.object_names)) # all ever added
        for i,oid in enumerate(oids):
            if not self.silent:
                print 'Working on "'+str(self.object_names[oid])+ \
                      '" (object', i+1, 'of', len(oids),')...'
            for f, visible in enumerate(self.object_visibility[oid]):
                if visible:
                    if not self.silent:
                        print '   Segmenting in frame '+str(f)+'...' 
                    self.segment_frame( oid, f )
                    self.object_volumes[oid][f] = self.netsurfs[oid][f].get_volume( pixelsize )
                    if not self.silent:
                        print '      Volume: ', self.object_volumes[oid][f]

    def segment_frame( self, oid, f ):
        """
        Segments object oid in frame f.
        """
        assert oid>=0
        assert oid<len(self.object_names)
        assert f>=0
        assert f<len(self.images)
        
        min_rs = (10,10,2)

        optimum = self.netsurfs[oid][f].apply_to(self.images[f], 
                                                 self.object_seedpoints[oid][f], 
                                                 self.object_max_surf_dist[oid], 
                                                 min_radii=self.object_min_surf_dist[oid])
        if not self.silent:
            print '      Optimum energy: ', optimum
            ins, outs = self.netsurfs[oid][f].get_counts()
            print '      Nodes in/out: ', ins, outs
            
    def get_center_estimates( self, oid, set_as_new=False ):
        better_centers = [None] * len(self.images)
        for f in range(len(self.images)):
            if not self.object_seedpoints[oid][f] is None:
                assert not self.netsurfs[oid][f] is None # segmentation must have been performed
                netsurf = self.netsurfs[oid][f]
                better_centers[f] = np.array(netsurf.get_surface_point(0))
                for i in range(1,netsurf.num_columns):
                    better_centers[f] += netsurf.get_surface_point(i)
                better_centers[f] /= netsurf.num_columns
                # better_centers[f] = list(better_centers[f])
        # update seedpoints if that was desired
        if set_as_new: self.object_seedpoints[oid] = better_centers
        return better_centers
            
    def show_frame( self, f, show_surfaces=False, show_centers=False, stackUnits=[1.,1.,1.], raise_window=True ):
        assert f>=0 and f<len(self.images)
        
        self.current_frame = f
        if self.spimagine is None:
            self.spimagine = volshow(self.images[f], stackUnits = stackUnits, raise_window=raise_window, autoscale=False)
        else:
            self.spimagine.glWidget.renderer.update_data(self.images[f])
            self.spimagine.glWidget.refresh()
        
        # remove all meshes (might eg exist from last call)
        self.hide_all_objects()
        
        for oid in range(len(self.object_names)):
            netsurf = self.netsurfs[oid][f]
            if not netsurf is None:
                if show_centers:  self.spimagine.glWidget.add_mesh( 
                        netsurf.create_center_mesh( facecolor=self.colors_diverse[0]) )
                if show_surfaces: self.spimagine.glWidget.add_mesh( 
                        netsurf.create_surface_mesh( facecolor=self.colors_diverse[0]) )
        return self.spimagine
    
    def hide_all_objects( self ):
        while len(self.spimagine.glWidget.meshes)>0:
            self.spimagine.glWidget.meshes.pop(0)
        self.spimagine.glWidget.refresh()

    def show_objects( self, oids, show_surfaces=True, show_centers=True, colors=None ):
        assert not self.current_frame is None
        if colors is None:
            colors = self.colors_diverse
            
        i = 0
        for oid in oids:
            assert oid>=0 and oid<len(self.object_names)
            netsurf = self.netsurfs[oid][self.current_frame]
            if not netsurf is None:
                if show_centers:  self.spimagine.glWidget.add_mesh( 
                        netsurf.create_center_mesh( facecolor=colors[i%len(colors)]) )
                if show_surfaces: self.spimagine.glWidget.add_mesh( 
                        netsurf.create_surface_mesh( facecolor=colors[i%len(colors)]) )
                i += 1
                
    def save_current_visualization( self, filename ):
        self.spimagine.saveFrame( filename )