public class Johnsons extends LLP {
    // You may be given inputs with negative weights, but no negative cost cycles
    // The graph may be directed, the weight of the edge from i to j is adjMatrix[i][j]
    int[] price;
    int[][] adjMatrix;
    int numVertices;
    public Johnsons(int[][] adjMatrix) {
        super(adjMatrix.length);
        numVertices = adjMatrix.length;
        this.price = new int[numVertices];
        this.adjMatrix = adjMatrix;
    }

    @Override
    public boolean forbidden(int j) {
        //Ensure that for every incoming edge, price[j] >= price[i] - edge[i, j]
        for(int i = 0; i < numVertices; i++){
            if(adjMatrix[i][j] != 0 /* Edge exists */ && price[j] < price[i] - adjMatrix[i][j] /* Edge condition */){
                return true;
            }
        }
        return false;
    }

    @Override
    public void advance(int j) {
        //Ensure that for every incoming edge, price[j] >= price[i] - edge[i, j]
        for(int i = 0; i < numVertices; i++){
            if(adjMatrix[i][j] != 0 /* Edge exists */ && price[j] < price[i] - adjMatrix[i][j] /* Edge condition */){
                //Print debug
                //System.out.println("price[" + j + "] advance to " + (price[i] - adjMatrix[i][j]));

                price[j] = price[i] - adjMatrix[i][j];
            }
        }
    }

    // This method will be called after solve()
    public int[] getSolution() {
        // Return the minimum price vector from Johnson's algorithm
        //Print debug
        //System.out.print("[ ");
        for(int i = 0; i < numVertices; i++){
            //System.out.print(price[i] + " ");
        }
        //System.out.println("]");

        return price;
    }
}