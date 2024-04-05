import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public abstract class LLP {
    // Feel free to add any methods here. Common parameters (e.g. number of processes)
    // can be passed up through a super constructor. Your code will be tested by creating
    // an instance of a sub-class, calling the solve() method below, and then calling the
    // sub-class's getSolution() method. You are free to modify anything else as long as
    // you follow this API (see SimpleTest.java)

    int numThreads;
    public LLP(int numThreads){
        this.numThreads = numThreads;
    }

    // Checks whether process j is forbidden in the state vector G.
    // If a thread is at a solution that is forbidden (not viable), it moves on to the next possible solution until something is found
    //
    public abstract boolean forbidden(int j);

    // Advances on process j
    public abstract void advance(int j);

    public void solve() {
        // Implement this method. There are many ways to do this but you
        // should follow the following basic steps:
        // 1. Compute the forbidden states
        // 2. Advance on forbidden states in parallel
        // 3. Repeat 1 and 2 until there are no forbidden states

        //Have a thread pool to give tasks to
        //Create a pool of processes and check if each is correct or not
        //For every thread it should keep checking if thread has forbidden solution, it should keep advancing if not forbidden
        //Don't need to be too careful with reading newest info, it just needs recheck if global state changes again (Thread can go from not forbidden to forbidden)
        //If global state changes, recheck every thread (Can be done in parallel).
        //Idea: Have a fence at the end, once all threads finish, check forbidden state for all threads again. Run again if forbidden
        //Implementation: Have a variable to determine if any forbidden == true occurred, if yes, run loop all over again until no forbidden occurs

        //Finding number of threads: length of G is number of threads you should have
        ExecutorService threadPool = Executors.newCachedThreadPool();

        //Keep running threads in parallel until a state is reached in which no threads call forbidden
        while(true) {
            //If stable state for an entire run, return
            boolean stable = true;

            //Return value from each thread
            ArrayList<Future<Boolean>> results = new ArrayList<>();

            //Create tasks
            for (int i = 0; i < numThreads; i++) {
                final int ID = i;

                //Create a new callable task for each thread with boolean return type
                Callable<Boolean> task = new Callable<Boolean>() {
                    //Override call method, this part is run when task is submitted
                    //Return true if forbidden is never called, else return false
                    @Override
                    public Boolean call() throws Exception {
                        //Debug
                        //System.out.println("Running task " + ID + " in thread " + Thread.currentThread().getName());

                        //Boolean for if forbidden is called
                        Boolean forbid = false;

                        //If forbidden, set forbid to true and keep advancing
                        while (forbidden(ID)) {
                            forbid = true;
                            advance(ID);
                        }
                        //Print debug
                        //if(forbid){ System.out.println("Running task " + ID + " in thread " + Thread.currentThread().getName() + "Forbids");}

                        //Return forbid status
                        return forbid;
                    }
                };

                //Get and add the future value to results array containing Future<Boolean> data
                Future<Boolean> future = threadPool.submit(task);
                results.add(future);
            }


            //Test if results contain false (forbidden called)
            for(int i = 0; i < results.size(); i++){
                //Obtain futures value (true = forbid)
                boolean threadResult;
                try{
                    //Gets each future object's boolean from list, will also block until object is submitted
                    threadResult = results.get(i).get();
                    //System.out.println("Thread " + i + " result: " + threadResult);
                } catch (Exception e) {
                    System.out.println("Thread exception");
                    throw new RuntimeException(e);
                }

                if(threadResult){
                    stable = false;
                }
            }

            //End loop if stable (no forbidden)
            if(stable){
                //System.out.println("End loop");
                break;
            }
        }

        //Loop finished, shut down thread pool
        threadPool.shutdown();
    }
}
