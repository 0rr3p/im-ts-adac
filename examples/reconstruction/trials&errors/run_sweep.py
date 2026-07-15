# run_sweep.py
import subprocess

pairs = [
    (66, 0.0),
    (63, 0.05),
    (58, 0.125),
    (52, 0.25),
    (37, 0.5),
]
joints = [1, 2, 3, 4, 5, 6]

for joint in joints:
    for epochs, overlap in pairs:
        subprocess.run([
            "python", "run_reconstruction.py", "-m",
            f"joint_id={joint}",
            f"training.num_epochs={epochs}",
            f"data.overlap={overlap}",
        ])