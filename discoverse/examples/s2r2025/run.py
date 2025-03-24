import subprocess

try:
    print("navigation to box")
    command = f"cd /workspace/DISCOVERSE/discoverse/examples/s2r2025 && python3 navigation_to_box.py"
    subprocess.run(command, shell=True, check=True)
except Exception as e:
    pass

try:
    print("ACT pick box")
    command = f"cd /workspace/DISCOVERSE/policies/act && python3 policy_evaluate_ros.py -tn mmk2_pick_box -ts 20250321-184740"
    subprocess.run(command, shell=True, check=True)
except Exception as e:
    pass

try:
    print("navigation to table")
    command = f"cd /workspace/DISCOVERSE/discoverse/examples/s2r2025 && python3 navigation_to_table.py"
    subprocess.run(command, shell=True, check=True)
except Exception as e:
    pass

try:
    print("ACT pick disk")
    command = f"cd /workspace/DISCOVERSE/policies/act && python3 policy_evaluate_ros.py -tn mmk2_pick_disk -ts 20250321-220431"
    subprocess.run(command, shell=True, check=True)
except Exception as e:
    pass