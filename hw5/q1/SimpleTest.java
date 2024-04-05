import org.junit.Test;
import static org.junit.Assert.*;

public class SimpleTest {

    final int[][] ADJ_MATRIX1 = {
      // 0,  1,  2,  3,  4,  5,  6,  7,  8 (vertex labels)
        {0,  0,  0,  0,  0,  0,  0,  1,  0}, // 0
        {0,  0,  7,  9,  0,  0,  14, 0,  0}, // 1
        {0,  7,  0,  10, 15, 0,  0,  0,  0}, // 2
        {0,  9,  10, 0,  11, 0,  2,  0,  0}, // 3
        {0,  0,  15, 11, 0,  6,  0,  0,  0}, // 4
        {0,  0,  0,  0,  6,  0,  9,  0,  0}, // 5
        {0,  14, 0,  2,  0,  9,  0,  0,  0}, // 6
        {1,  0,  0,  0,  0,  0,  0,  0,  7}, // 7
        {0,  0,  0,  0,  0,  0,  0,  7,  0}  // 8
    };
    final int[][] ADJ_MATRIX2 = {
      // 0, 1, 2, 3 (vertex labels)
        {0, -5, 2, 3}, // 0
        {0, 0, 4, 0}, // 1
        {0, 0, 0, 1}, // 2
        {0, 0, 0, 0}
    };

    final int[] COMPONENTS = {8, 6, 6, 6, 6, 6, 6, 8, 8};
    final int[] SOURCE1_SPATH_COSTS = {Integer.MAX_VALUE, 0, 7, 9, 20, 20, 11, Integer.MAX_VALUE, Integer.MAX_VALUE};
    final int[] JOHNSON_PRICES = {0, 5, 1, 0};
    
    @Test
    public void testConnectedComponents() {
        ConnectedComponents cc = new ConnectedComponents(ADJ_MATRIX1);
        cc.solve();
        int[] components = cc.getSolution();
        assertArrayEquals(components, COMPONENTS);
    }

    @Test
    public void testBellmanFord() {
        int source = 1;
        BellmanFord bf = new BellmanFord(ADJ_MATRIX1, source);
        bf.solve();
        int[] costs = bf.getSolution();
        assertArrayEquals(costs, SOURCE1_SPATH_COSTS);
    }
    
    @Test
    public void testJohnsons() {
        Johnsons johnsons = new Johnsons(ADJ_MATRIX2);
        johnsons.solve();
        int[] prices = johnsons.getSolution();
        assertArrayEquals(prices, JOHNSON_PRICES);
    }
}
