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

