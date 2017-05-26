class Fiducials2d:
    """
    Class that helps to organize and track history of fiducial markers.
    """

    next_id = 0
    fiducials = None
    history = None

    def __init__( self ):
        """
        Parameters:
        """
        self.fiducials = {}
        self.history = {}

    def get_ids(self):
        return self.fiducials.keys()

    def get(self, id):
        return self.fiducials[id]

    def get_history(self, id):
        return self.history[id]

    def add_fiducial(self, point):
        '''
        :param point: coordinate of fiducial to be added (x,y) 
        :return: the id of the added fiducial marker
        '''
        self.fiducials[self.next_id] = point
        self.history[self.next_id] = []
        self.next_id += 1
        return self.next_id-1

    def add_fiducials(self, points):
        '''
        :param points: list of points to be added (list of tuples of (x,y)-coordinates)
        :return: list of ids created by adding given points
        '''
        ids = []
        for point in points:
            ids.append(self.add_fiducial(point))
        return ids

    def remove_fiducial(self, id):
        '''
        :param id: id of fiducial to be deleted
        :return: 
        '''
        if self.fiducials.has_key(id):
            del self.fiducials[id]
            del self.history[id]

    def remove_fiducials(self, ids):
        '''
        :param ids: list of fiducial ids to be removed 
        :return: 
        '''
        for id in ids:
            self.remove_fiducial(id)

    def move(self, id, new_position, add_to_history=True):
        '''
        Moves a fiducial marker
        :param id: id of the fiducial to be moved
        :param new_position: (x,y)-tuple of updated position
        :param add_to_history: if True, current position will be added to position history
        :return: 
        '''
        assert self.fiducials.has_key(id)

        if add_to_history:
            self.history[id].insert(0, self.fiducials[id])

        self.fiducials[id] = new_position

    def reset(self, id, point):
        '''
        Resets a fiducial marker. (This clears hi history!)
        :param id: id of the fiducial to be reset
        :param point: the new position
        :return: 
        '''
        assert self.fiducials.has_key(id)

        self.remove_fiducial(id)

        self.fiducials[id] = point
        self.history[id] = []
