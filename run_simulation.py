# run_simulation.py

import sys
import os
import time
import dill
from  natsort import humansorted

# Add the src directory to the system path so it can find the modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import your classes and utilities
from src.eptm_class import Epithelium
from src.logging_utils import logger,clear_log_file




# Global variable to keep track of the number of evolution steps
t = 0
dt = 0.002*0.5

evolution_limit = 4000  # Limit the number of dynamic evolution steps to 3

# Set the delay duration in seconds (e.g., 0.5 seconds)
frame_delay = 0# 0.00001

def run_simulation():
    clear_log_file()  # Ensure the log file is cleared at the start of the simulation
    save_dir = '/home/ardati/PycharmProjects/pickled_toygl_tissues'
    global t, dt, t_div_elem, last_interval_index,save_steps,Data_dir
    while eptm.step < evolution_limit:
        t += dt  # Advance time
        logger.superior_info(f"At step : {eptm.step},"
                             f" Number of facets :{len(eptm.cells[-1].facets)},"
                             f" filaments:{len(eptm.filaments)},"
                             f" Nodes:{len(eptm.nodes)}")
        logger.dont_debug(f"At step : {eptm.step}")
        eptm.dynamic_evolution(t, dt)  # Dynamic evolution of the epithelium
        eptm.model_eptm_behaviour(t)
        # Additional evolution functions can be added here
        # Save the advancement
        eptm.step += 1
        eptm.fast_export_facets(output_dir=f'{Data_dir}/fastoutputfacets/', step=eptm.step)

        if save_steps:
            eptm.pickle_self(SAVE_DIR=Data_dir,name=f"step_{eptm.step}")
    logger.info_once("Evolution limit reached.")
    logger.superior_info(f"At step : {eptm.step},"
                         f" Number of facets :{len(eptm.cells[-1].facets)},"
                         f" filaments:{len(eptm.filaments)},"
                         f" Nodes:{len(eptm.nodes)}")
    # display()

    # Introduce a delay between frames
    # time.sleep(frame_delay)




if __name__ == '__main__':
    # Define the test name
    test_name = "Test_01_Dec"

    # Start timing
    start_time = time.time()

    Save_dir = "/home/ardati/Data_ToyGL/"
    Data_dir = "/home/ardati/PycharmProjects/Data_ToyGL"

    sim_name = test_name
    Save_dir = Save_dir + sim_name
    print(f"Save dir : {Save_dir}")

    # Initialize an epithelium
    eptm = Epithelium()
    Recup = False
    save_steps = False

    if Recup:
        print(f"## Loading a toygl tissue ##...  ")
        Save_dir = "/home/ardati/PycharmProjects/pickled_toygl_tissues"
        tissues = humansorted(os.listdir(Save_dir))
        tissu_name = tissues[-1]
        # or choose a specific tissue to loop from
        tissu_name = 'step_880'

        print(f"tissues : {tissues}")
        open_dir = os.path.join(f"{Save_dir}", f"{tissu_name}")
        with open(open_dir, 'rb') as s:
            eptm = dill.load(s)
            print(f"Successfully loaded Relaxed Voronoi tissue: {tissu_name}")

        print(f"step : {eptm.step}")
        step = eptm.step
    else:
        # Gcells = eptm.create_an_eptm_of_two_growing_cells()
        Gcells = eptm.create_an_eptm_of_a_growing_cells()

        cellA = eptm.cells[-1]

    # Run the simulation
    run_simulation()

    # End timing
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken to execute the whole code: {elapsed_time:.4f} seconds")

    # Save the timing information to a text file
    timing_file_path = "/home/ardati/PycharmProjects/toygl/simulation_times.txt"
    with open(timing_file_path, "a") as f:
        f.write(f"{test_name}: {elapsed_time:.4f} seconds\n")
    print(f"Timing information saved to {timing_file_path}")







