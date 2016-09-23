import numpy as np
import bresenham as bham
import maxflow



def compute_weight( img, coords ):
    '''
    img     numpy array containing the image data
    coords  list of lists containing as many entries as img has dimensions
    '''
    m = 0
    for c in coords:
        try:
            m = max( m,img[ tuple(c[::-1]) ] ) # [::-1]
        except:
            None
    return m

def compute_weights( image, center, K, max_radii, col_vectors, inverse_order=False, min_radii=[0,0,0] ):
    '''
    Computes all weights of G and of G_tilde and returns them as a tuple (w, w_tilde).
    Parameters:
    - image         the image data (3d numpy array)
    - center        tupel with center coordinates (x, y, z)
    - K             number of nodes along each col_vector
    - max_radii     tupel containing number of pixels in (x, y, z) directions
    - col_vectors   vectors pointing on the unit sphere in all directions to be sampled
    - inverse_order if true, the computed weights will be sorted such that they go from
                    the outside to the inside of the given image (along same vectors)
    - min_radii     defines the min_radii to start sampling the col_vectors at
    '''
    num_columns = len(col_vectors)
    w = np.zeros([num_columns, K]) # node weights
    w_tilde = np.zeros([num_columns, K])

    # fill in node weights
    for i in range(num_columns):
        from_x = int(center[0] + col_vectors[i,0]*min_radii[0])
        from_y = int(center[1] + col_vectors[i,1]*min_radii[1])
        from_z = int(center[2] + col_vectors[i,2]*min_radii[2])
        to_x = int(center[0] + col_vectors[i,0]*max_radii[0])
        to_y = int(center[1] + col_vectors[i,1]*max_radii[1])
        to_z = int(center[2] + col_vectors[i,2]*max_radii[2])
        coords = bham.bresenhamline(np.array([[from_x, from_y, from_z]]), np.array([[to_x, to_y, to_z]]))
        num_pixels = len(coords)
        for k in range(K):
            start = int(k * float(num_pixels)/K)
            end = max( start+1, start + num_pixels/K )
            w[i,k] = -1 * compute_weight( image, coords[start:end])

    if inverse_order:
        w = w[:,::-1]

    for i in range(num_columns):
        w_tilde[i,0] = w[i,0] 
        for k in range(1,K):
            w_tilde[i,k] = w[i,k]-w[i,k-1]
            
    return w, w_tilde

def build_flow_network( num_columns, neighbors_of, K, num_neighbors, max_delta_k, w_tilde, alpha=None ):
    '''
    Builds the flow network that can solve the V-Weight Net Surface Problem
    Returns a tuple (g, nodes) consisting of the flow network g, and its nodes.
    '''
    INF = 10000000

    num_nodes = num_columns*K
    num_edges = (num_nodes*num_neighbors*(max_delta_k+max_delta_k+1))/2

    g = maxflow.Graph[float](num_nodes,num_edges)
    nodes = g.add_nodes(num_nodes)

    for i in range(num_columns):

        # connect column to s,t
        for k in range(K):
            if w_tilde[i,k] < 0:
                g.add_tedge(i*K+k, -w_tilde[i,k], 0)
            else:
                g.add_tedge(i*K+k, 0, w_tilde[i,k])

        # connect column to i-chain
        for k in range(1,K):
            g.add_edge(i*K+k, i*K+k-1, INF, 0)

        # connect column to neighbors
        for k in range(K):
            for j in neighbors_of[i]:
                k2 = max(0,k-max_delta_k)
                g.add_edge(i*K+k, j*K+k2, INF, 0)
                if alpha != None:
                    # add constant cost penalty \alpha
                    g.add_edge(i*K+k, j*K+k, alpha, 0)

    return g, nodes