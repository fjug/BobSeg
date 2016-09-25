import numpy as np
import bresenham as bham
import maxflow
import math

from spimagine import EllipsoidMesh, Mesh

class NetSurf3d:
    """
    Implements a 3d version of the optimal net surface problem.
    Relevant publication: [Wu & Chen 2002]
    """
    
    INF = 99999999
    
    image = None
    center = None
    min_radii = None
    max_radii = None
    
    w = None
    w_tilde = None
    
    nodes = None
    edges = None
    g = None
    maxval = None
    
    def __init__( self, columns, triangles, adjacency, K=30, max_delta_k=4 ):
        """
        Parameters:
            columns     -  unit vectors defining the direction of net columns
            triangles   -  list of column id triplets defining the faces of the 3d object definied by columns
            adjancency  -  neighborhood of columns to be used in net surface
            K           -  how many sample points per column
            max_delta_k -  maximum column height change between neighbors (as defined by adjacency)
        """
        assert (columns.shape[1] == 3)
        
        self.col_vectors = columns
        self.triangles = triangles
        self.neighbors_of = adjacency
        self.K = K
        self.max_delta_k = max_delta_k
        
        self.num_columns = len(columns)

    def apply_to( self, image, center, max_radii, min_radii=(0,0,0) ):
        assert( len(image.shape) == 3 )
        assert( len(center) == 3 )
        assert( len(max_radii) == 3 )
        assert( len(min_radii) == 3 )
    
        self.image = image
        self.center = np.array(center)
        self.min_radii = min_radii
        self.max_radii = max_radii
        
        self.compute_weights()
        self.build_flow_network()
        
        self.maxval = self.g.maxflow()
        return self.maxval
        
    def compute_weights(self, inverse_order=False):
        '''
        Computes all weights of G and of G_tilde and returns them as a tuple (w, w_tilde).
        '''
        
        assert not self.image is None
        
        self.w = np.zeros([self.num_columns, self.K]) # node weights
        self.w_tilde = np.zeros([self.num_columns, self.K])

        # fill in node weights
        for i in range(self.num_columns):
            from_x = int(self.center[0] + self.col_vectors[i,0]*self.min_radii[0])
            from_y = int(self.center[1] + self.col_vectors[i,1]*self.min_radii[1])
            from_z = int(self.center[2] + self.col_vectors[i,2]*self.min_radii[2])
            to_x = int(self.center[0] + self.col_vectors[i,0]*self.max_radii[0])
            to_y = int(self.center[1] + self.col_vectors[i,1]*self.max_radii[1])
            to_z = int(self.center[2] + self.col_vectors[i,2]*self.max_radii[2])
            coords = bham.bresenhamline(np.array([[from_x, from_y, from_z]]), np.array([[to_x, to_y, to_z]]))
            num_pixels = len(coords)
            for k in range(self.K):
                start = int(k * float(num_pixels)/self.K)
                end = max( start+1, start + num_pixels/self.K )
                self.w[i,k] = -1 * self.compute_weight_at(coords[start:end])

        if inverse_order:
            self.w = self.w[:,::-1]

        for i in range(self.num_columns):
            self.w_tilde[i,0] = self.w[i,0] 
            for k in range(1,self.K):
                self.w_tilde[i,k] = self.w[i,k]-self.w[i,k-1]

    def compute_weight_at( self, coords ):
        '''
        coords  list of lists containing as many entries as img has dimensions
        '''
        m = 0
        for c in coords:
            try:
                m = max( m,self.image[ tuple(c[::-1]) ] )
            except:
                None
        return m

    def build_flow_network( self, alpha=0.0 ):
        '''
        Builds the flow network that can solve the V-Weight Net Surface Problem
        Returns a tuple (g, nodes) consisting of the flow network g, and its nodes.
        
        If alpha != 0.0 this method will add an additional weighted flow edge (horizontal binary costs).
        '''
        self.num_nodes = self.num_columns*self.K
        # estimated num edges (in case I'd have equal num enighbors and full pencils)
        self.num_edges = ( self.num_nodes *
                           len(self.neighbors_of[0]) * 
                           (self.max_delta_k + self.max_delta_k+1) ) * .5

        self.g = maxflow.Graph[float]( self.num_nodes, self.num_edges)
        self.nodes = self.g.add_nodes( self.num_nodes )

        for i in range( self.num_columns ):

            # connect column to s,t
            for k in range( self.K ):
                if self.w_tilde[i,k] < 0:
                    self.g.add_tedge(i*self.K+k, -self.w_tilde[i,k], 0)
                else:
                    self.g.add_tedge(i*self.K+k, 0, self.w_tilde[i,k])

            # connect column to i-chain
            for k in range(1,self.K):
                self.g.add_edge(i*self.K+k, i*self.K+k-1, self.INF, 0)

            # connect column to neighbors
            for k in range(self.K):
                for j in self.neighbors_of[i]:
                    k2 = max(0,k-self.max_delta_k)
                    self.g.add_edge(i*self.K+k, j*self.K+k2, self.INF, 0)
                    if alpha != None:
                        # add constant cost penalty \alpha
                        self.g.add_edge(i*self.K+k, j*self.K+k, alpha, 0)
                        
    def get_counts( self ):
        size_s_comp = 0
        size_t_comp = 0
        for n in self.nodes:
            seg = self.g.get_segment(n)
            if seg == 0:
                size_s_comp += 1
            else:
                size_t_comp += 1
        return size_s_comp, size_t_comp
    
    
    def norm_coords(self,cabs,pixelsizes):
        """ 
        converts from absolute pixel location in image (x,y,z) to normalized [0,1] coordinates for spimagine meshes (z,y,x).
        """        
        cnorm = 2. * np.array(cabs[::-1], float) / np.array(pixelsizes) - 1.
        return tuple(cnorm[::-1])

    def norm_radii(self,cabs,pixelsizes):
        """ 
        converts from absolute pixel based radii to normalized [0,1] coordinates for spimagine meshes (z,y,x).
        """        
        cnorm = 2. * np.array(cabs[::-1], float) / np.array(pixelsizes)
        return tuple(cnorm[::-1])

    def create_center_mesh( self, facecolor=(1.,.3,.2), radii=min_radii ):
        
         if radii is None: radii = (3,3,.5)
         return EllipsoidMesh(rs=self.norm_radii(radii,self.image.shape), 
                              pos=self.norm_coords(self.center, self.image.shape), 
                              facecolor=facecolor, 
                              alpha=.5)
    
    def create_surface_mesh( self, facecolor=(1.,.3,.2) ):
        myverts = np.zeros((self.num_columns, 3))
        mynormals = self.col_vectors
        
        for i in range(self.num_columns):
            p = self.get_surface_point(i)
            myverts[i,:] = self.norm_coords( p, self.image.shape )
                
        return Mesh(vertices=myverts, normals = mynormals, indices = self.triangles.flatten(), facecolor=facecolor, alpha=.5)
    
    def get_volume( self, calibration = (1.,1.,1.) ):
        """
        calibration: 3-tupel of pixel size multipliers
        """
        volume = 0.
        for a,b,c in self.triangles:
            pa = self.get_surface_point( a )
            pb = self.get_surface_point( b )
            pc = self.get_surface_point( c )    
            volume += self.get_triangle_splinter_volume( pa, pb, pc, calibration )

        return volume   
         
            
    def get_surface_point( self, column_id ):
        for k in range(self.K):
            if self.g.get_segment(column_id*self.K+k) == 1: break # leave as soon as k is first outside point
        k-=1
        x = int(self.center[0] + self.col_vectors[column_id,0] * 
                self.min_radii[0] + self.col_vectors[column_id,0] * 
                (k-1)/float(self.K) * (self.max_radii[0]-self.min_radii[0]) )
        y = int(self.center[1] + self.col_vectors[column_id,1] * 
                self.min_radii[1] + self.col_vectors[column_id,1] * 
                (k-1)/float(self.K) * (self.max_radii[1]-self.min_radii[1]) )
        z = int(self.center[2] + self.col_vectors[column_id,2] * 
                self.min_radii[2] + self.col_vectors[column_id,2] * 
                (k-1)/float(self.K) * (self.max_radii[2]-self.min_radii[2]))
        return (x,y,z)
    
    def get_triangle_splinter_volume( self, pa, pb, pc, calibration ):
        """
        Computes the volume of the pyramid defined by points pa, pb, pc, and self.center
        """
        assert not self.center is None
        
        x = (np.array(pa)-self.center) * calibration[0]
        y = (np.array(pb)-self.center) * calibration[1]
        z = (np.array(pc)-self.center) * calibration[2]
        return math.fabs( x[0] * y[1] * z[2] + 
                          x[1] * y[2] * z[0] + 
                          x[2] * y[0] * z[1] - 
                          x[0] * y[2] * z[1] - 
                          x[1] * y[0] * z[2] - 
                          x[2] * y[1] * z[0]) / 6.