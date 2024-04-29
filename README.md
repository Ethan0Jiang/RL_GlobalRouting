# Global Routing with Reinforcement Learning (RL)

Global routing is a critical step in Very Large Scale Integration (VLSI) design, directly influencing circuit timing, power consumption, and overall routability. Traditional global routing methods often rely on greedy algorithms and predefined heuristics, which can lead to inefficiencies and suboptimal solutions. These limitations can impact circuit performance and scalability.

To overcome these challenges, this project proposes a reinforcement learning (RL) approach to global routing. Our method is designed to be more adaptable, allowing for a dynamic response to complex routing scenarios. This project incorporates two advanced RL techniques: Monte Carlo Tree Search (MCTS) and Deep Q-Networks (DQN). These methods demonstrate how RL can be a viable alternative to conventional global routing techniques in VLSI design, aiming for better routing efficiency and circuit performance.

## Project Structure

Here's an overview of the key files and directories in this repository:

- **/GlobalRoutingRL/BenchmarkGenerator**: Contains the benchmark generator for creating test cases for the RL algorithm.
- **/GlobalRoutingRL/DQN_model_v3.py**: Defines the DQN agent, used for reinforcement learning.
- **/GlobalRoutingRL/MCTS_model.py**: Implements the MCTS agent, used for reinforcement learning.
- **/GlobalRoutingRL/RoutingEnv_v2.py**: The final version of the reinforcement learning environment for global routing.
- **/GlobalRoutingRL/MST.py**: A utility for decomposing multi-pin nets into two-pin nets, used for preprocessing.
- **/GlobalRoutingRL/benchmark**: Contains all the benchmark files used for testing and evaluating the RL agents.
- **/GlobalRoutingRL/output**: The output folder where results are stored after running benchmarks.

## Running the Code

To run the project, you can use either the provided bash script or individual Python commands.

### Using the Bash Script
1. Open a terminal and navigate to the project directory:
   ```bash
   cd ./GlobalRoutingRL
   ```
2. Grant execute permissions to the bash script:
   ```bash
   chmod +x run_benchmarks.sh
   ```
3. Run the script to execute all benchmarks:
   ```bash
   ./run_benchmarks.sh
   ```

### Using Individual Python Commands
If you prefer, you can run the project with individual Python commands:
1. Open a terminal and navigate to the project directory:
   ```bash
   cd ./GlobalRoutingRL
   ```
2. Execute the main script with specific benchmarks and specify the output files:
   ```bash
   python main.py benchmark/test_benchmark_1.gr output/bm1_result.txt
   python main.py benchmark/test_benchmark_2.gr output/bm2_result.txt
   python main.py benchmark/test_benchmark_3.gr output/bm3_result.txt
   python main.py benchmark/test_benchmark_4.gr output/bm4_result.txt
   python main.py benchmark/test_benchmark_5.gr output/bm5_result.txt
   ```
3. To evaluate the MCTS, user can run:
   ```bash
   python eval_mcts.py benchmark/test_benchmark_5.gr output/mcts_eval/bm5_mcts_result.txt 10,20,50,100 2,5,10,20
   ```


These commands generate output files for each benchmark, providing insights into routing performance and results.

