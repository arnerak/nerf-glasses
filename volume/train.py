import sys
import os

# import Instant-NGP python bindings
INGP_ROOT = "/opt/instant-ngp"
PYNGP_PATH = INGP_ROOT + "/build"
sys.path.append(PYNGP_PATH)
import pyngp as ngp

# stop training when target loss is achieved or after a max number of training steps
TARGET_LOSS = 0.00175
MAX_TRAINING_STEPS = 10000

if __name__ == "__main__":
    dataset_path = sys.argv[1]

    testbed = ngp.Testbed()
    testbed.root_dir = INGP_ROOT
    testbed.load_training_data(sys.argv[1])
    testbed.shall_train = True

    while testbed.frame():
        if testbed.loss < TARGET_LOSS or testbed.training_step >= MAX_TRAINING_STEPS:
            break
        print("\rTraining step: ", testbed.training_step, end="")
    print("\nTraining complete with loss ", testbed.loss)
    
    snapshot_path = dataset_path
    if not os.path.isdir(snapshot_path):
        snapshot_path = os.path.dirname(snapshot_path)
    snapshot_path = os.path.join(snapshot_path, "nerf.msgpack")

    testbed.save_snapshot(snapshot_path)        

