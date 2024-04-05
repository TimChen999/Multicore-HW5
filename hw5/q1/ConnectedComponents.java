public class ConnectedComponents extends LLP {
    int numVertices;
    int[][] adjMatrix;
    int adjacency[];
    public ConnectedComponents(int[][] adjMatrix) {
        super(adjMatrix.length);
        this.numVertices = adjMatrix.length;
        this.adjMatrix = adjMatrix;
        this.adjacency = new int[numVertices];
        for(int i = 0; i < numVertices; i++){
            adjacency[i] = i;
        }
        //Print debug
        //System.out.print("[ ");
        for(int i = 0; i < numVertices; i++){
            //System.out.print(adjacency[i] + " ");
        }
        //System.out.println("]");
    }

    @Override
    public boolean forbidden(int j) {
        //Find max adjacent
        for(int i = 0; i < numVertices; i++){
            //Max connected (undirected)
            if((adjMatrix[i][j] != 0 || adjMatrix[j][i] != 0) && adjacency[i] > adjacency[j]){
                return true;
            }
        }
        return false;
    }

    @Override
    public void advance(int j) {
        //Set max adjacent
        for(int i = 0; i < numVertices; i++){
            //Max connected (undirected)
            if((adjMatrix[i][j] != 0 || adjMatrix[j][i] != 0) && adjacency[i] > adjacency[j]){
                adjacency[j] = adjacency[i];

                //Print debug
                //System.out.println("Set index: " + j + " to " + i);
            }
        }
    }

    // This method will be called after solve()
    public int[] getSolution() {
        // Return the vector where the i^th entry is the index j where
        // j is the largest vertex label contained in the component containing 
        // vertex i

        //Print debug
        //System.out.print("[ ");
        for(int i = 0; i < numVertices; i++){
            //System.out.print(adjacency[i] + " ");
        }
        //System.out.println("]");
        return adjacency;
    }
}
