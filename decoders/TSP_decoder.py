class TSPDecoder():
    """
    Simple Traveling Salesman Problem decoder. It creates a permutation of
    nodes induced by the chromosome and computes the cost of the tour.
    """

    def __init__(self, instance):
        self.instance = instance

    ###########################################################################

    def decode(self, chromosome) -> float:
        """
        Given a chromossome, builds a tour.

        Note that in this example, ``rewrite`` has not been used.
        """

        permutation = sorted(
            (key, index) for index, key in enumerate(chromosome)
        )

        cost = self.instance.distance(permutation[0][1], permutation[-1][1])
        for i in range(len(permutation) - 1):
            cost += self.instance.distance(permutation[i][1],
                                           permutation[i + 1][1])
        return cost