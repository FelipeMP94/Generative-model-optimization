class TSPInstance():
   

    def __init__(self, filename: str):
        """
        Initializes the instance loading from a file.
        """

        with open(filename, "r") as hd:
            lines = hd.readlines()


        line_number = 1
        
        self.num_nodes = int(lines[0])

        matrix_size = (self.num_nodes * (self.num_nodes - 1)) / 2
        self.distances = []

        for i in range(1, self.num_nodes):
            line_number = i + 1
            values = [float(x.strip()) for x in lines[i].split()]
            self.distances.extend(values)
    

    ###########################################################################

    def distance(self, i: int, j: int) -> float:
        """
        Returns the distance between nodes `i` and `j`.
        """
        if i > j:
            i, j = j, i
        return self.distances[(i * (self.num_nodes - 1)) - ((i - 1) * i // 2) +
                              (j - i - 1)]