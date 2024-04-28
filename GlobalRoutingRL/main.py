import sys
import os
import DQN_model_v3
import MCTS_model

def run_router(input_file_path, output_file_path):

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    f = open(output_file_path, "w")

    f.write("Benchmark:  {}\n".format(input_file_path))

    print("/////// Running DQN-based router ///////")
    f.write("\nDQN-based router\n")
    total_congestion, min_capacity, total_wire_length = DQN_model_v3.solve_routing_with_dqn(input_file_path)
    print("Total congestion:", total_congestion)
    print("Minimum capacity:", min_capacity)
    print("Total wire length:", total_wire_length)
    f.write("Total congestion: {}\n".format(total_congestion))
    f.write("Minimum capacity: {}\n".format(min_capacity))
    f.write("Total wire length: {}\n".format(total_wire_length))

    print("/////// Running MCTS-based router ///////")
    f.write("\nMCTS-based router\n")
    total_congestion, min_capacity, total_wire_length = MCTS_model.solve_routing_with_mcts(input_file_path)
    print("Total congestion:", total_congestion)
    print("Minimum capacity:", min_capacity)
    print("Total wire length:", total_wire_length)
    f.write("Total congestion: {}\n".format(total_congestion))
    f.write("Minimum capacity: {}\n".format(min_capacity))
    f.write("Total wire length: {}\n".format(total_wire_length))

    f.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py input_file_path output_file_path")
        sys.exit(1)

    # input_file_path = 'benchmark/test_benchmark_1.gr'
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    run_router(input_file_path, output_file_path)
