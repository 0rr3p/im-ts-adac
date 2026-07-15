import subprocess

joints = [1, 2, 3, 4, 5, 6]

for joint in joints:
    subprocess.run(["python", "ad.py", f"joint_id={joint}",])
        