import sys
import os
import time  # For measuring runtime
import DQN_model_v3
import MCTS_model

def run_router(input_file_path, output_file_path):
    # Ensure the output directory exists
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, "w") as f:
        f.write("Benchmark: {}\n".format(input_file_path))

        print("/////// Running DQN-based router ///////")
        f.write("\nDQN-based router\n")

        # Record start time
        start_time = time.time()

        # Add runtime for DQN-based router
        total_congestion, min_capacity, total_wire_length = DQN_model_v3.solve_routing_with_dqn(input_file_path)
        
        # Calculate elapsed time
        dqn_runtime = time.time() - start_time

        print("Total congestion:", total_congestion)
        print("Minimum capacity:", min_capacity)
        print("Total wire length:", total_wire_length)
        print("Runtime:", dqn_runtime, "seconds")

        f.write("Total congestion: {}\n".format(total_congestion))
        f.write("Minimum capacity: {}\n".format(min_capacity))
        f.write("Total wire length: {}\n".format(total_wire_length))
        f.write("Runtime: {} seconds\n".format(dqn_runtime))

        print("/////// Running MCTS-based router ///////")
        f.write("\nMCTS-based router\n")

        # Record start time for MCTS-based router
        start_time = time.time()

        # Add runtime for MCTS-based router
        total_congestion, min_capacity, total_wire_length = MCTS_model.solve_routing_with_mcts(input_file_path)

        # Calculate elapsed time for MCTS
        mcts_runtime = time.time() - start_time

        print("Total congestion:", total_congestion)
        print("Minimum capacity:", min_capacity)
        print("Total wire length:", total_wire_length)
        print("Runtime:", mcts_runtime, "seconds")

        f.write("Total congestion: {}\n".format(total_congestion))
        f.write("Minimum capacity: {}\n".format(min_capacity))
        f.write("Total wire length: {}\n".format(total_wire_length))
        f.write("Runtime: {} seconds\n".format(mcts_runtime))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py input_file_path output_file_path")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    run_router(input_file_path, output_file_path)