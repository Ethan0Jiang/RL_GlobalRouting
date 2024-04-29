import sys
import os
import time
import pandas as pd  # For creating dataframes and tables
import DQN_model_v3
import MCTS_model


def eval_mcts(input_file_path, output_file_path, simulations, actions):
    # Ensure the output directory exists
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    results = []  # Store results for table creation

    with open(output_file_path, "w") as f:
        f.write("Benchmark: {}\n".format(input_file_path))

        # Loop through the simulations and actions to sweep configurations
        for sim in simulations:
            for act in actions:
                f.write("\nConfiguration: Simulations={} | Actions={}\n".format(sim, act))
                print("Running MCTS-based router with Simulations:", sim, "Actions:", act)

                # Measure the runtime for each configuration
                start_time = time.time()

                # Get results from MCTS-based router
                total_congestion, min_capacity, total_wire_length = MCTS_model.solve_routing_with_mcts(input_file_path, sim, act)

                # Calculate elapsed time for MCTS
                mcts_runtime = time.time() - start_time

                # Print the results
                print("Total congestion:", total_congestion)
                print("Minimum capacity:", min_capacity)
                print("Total wire length:", total_wire_length)
                print("Runtime:", mcts_runtime, "seconds")

                # Store the results for creating tables
                results.append({
                    "simulations": sim,
                    "actions": act,
                    "congestion": total_congestion,
                    "capacity": min_capacity,
                    "wire_length": total_wire_length,
                    "runtime": mcts_runtime,
                })

    # Create tables for each metric and save to LaTex-friendly format
    create_tables(results, output_file_path)


def create_tables(results, output_file_path):
    # Convert results into a dataframe for easier manipulation
    df = pd.DataFrame(results)

    # Pivot the data to create a table for Total Congestion
    congestion_table = df.pivot(index="simulations", columns="actions", values="congestion")
    congestion_table.to_csv(output_file_path.replace(".txt", "_congestion.csv"), index=True)

    # Pivot the data to create a table for Minimum Capacity
    capacity_table = df.pivot(index="simulations", columns="actions", values="capacity")
    capacity_table.to_csv(output_file_path.replace(".txt", "_capacity.csv"), index=True)

    # Pivot the data to create a table for Total Wire Length
    wire_length_table = df.pivot(index="simulations", columns="actions", values="wire_length")
    wire_length_table.to_csv(output_file_path.replace(".txt", "_wire_length.csv"), index=True)

    # Pivot the data to create a table for Runtime
    runtime_table = df.pivot(index="simulations", columns="actions", values="runtime")
    runtime_table.to_csv(output_file_path.replace(".txt", "_runtime.csv"), index=True)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script_name.py input_file_path output_file_path num_simulations num_actions")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    num_simulations = list(map(int, sys.argv[3].split(',')))
    num_actions = list(map(int, sys.argv[4].split(',')))

    eval_mcts(input_file_path, output_file_path, num_simulations, num_actions)
