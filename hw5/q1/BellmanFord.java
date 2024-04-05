public class BellmanFord extends LLP {
    // You may be given inputs with negative weights, but no negative cost cycles
    // The graph may be directed, the weight of the edge from i to j is adjMatrix[i][j]
    int[][] adjMatrix;
    int[] distanceToSource;
    int source;
    int numVertices;
    public BellmanFord(int[][] adjMatrix, int source) {
        super(adjMatrix.length /*Number of vertices*/);
        this.adjMatrix = adjMatrix;
        this.source = source;
        this.numVertices = adjMatrix.length;
        distanceToSource = new int[numVertices];
        //Init array
        for(int i = 0; i < numVertices; i++){
            if(i != source){
                distanceToSource[i] = Integer.MAX_VALUE;
            }
        }
        int x = 0;
    }

    @Override
    public boolean forbidden(int j) {
        //System.out.println("Dist j (" + j + ") is: " + distanceToSource[j]);
        for(int i = 0; i < numVertices; i++){
            //For all incoming edges to j, [i][j] == valid int
            //Check if distance can be improved from i and edge [i][j]
            if(i != j /* Don't compare to self */ && distanceToSource[i] != Integer.MAX_VALUE /* Node i has been reached */ && adjMatrix[i][j] != 0 /* Edge exists */ && distanceToSource[i] + adjMatrix[i][j] < distanceToSource[j] /* New shortest edge */) {
                //System.out.println("\tDist j (" + j + ") from i (" + i + "), given edge weight: " + adjMatrix[i][j] + " is: " + (distanceToSource[i] + adjMatrix[i][j]));
                return true;
            }
        }
        return false;
    }

    @Override
    public void advance(int j) {
        for(int i = 0; i < numVertices; i++){
            //Update distanceToSource[j] to shortest distance
            if(i != j /* Don't compare to self */ && distanceToSource[i] != Integer.MAX_VALUE /* Node i has been reached */ && adjMatrix[i][j] != 0 /* Edge exists */ && distanceToSource[i] + adjMatrix[i][j] < distanceToSource[j] /* New shortest edge */) {
                //System.out.println("Advance: Set new " + j + " via edge " + i + " to " + j);
                distanceToSource[j] = distanceToSource[i] + adjMatrix[i][j];
            }
        }
    }

    // This method will be called after solve()
    public int[] getSolution() {
        // Return the vector of shortest path costs from source to each vertex
        // If a vertex is not connected to the source then its cost is Integer.MAX_VALUE
        return distanceToSource;
    }
}
