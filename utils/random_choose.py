import random

input_files = [
    "guba_1_1000.txt",
    "guba_1000_2000.txt",
    "guba_2000_3000.txt",
    "guba_3000_4000.txt",
    "guba_4000_5000.txt",
    "guba_5000_6000.txt"
]

all_lines = []
for file_name in input_files:
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            all_lines.extend([line for line in lines if line.strip()])
    except Exception as e:
        print(f"Error reading {file_name}: {e}")

if len(all_lines) < 128:
    raise ValueError("Not enough data to sample 128 lines.")

random.seed(42)
sampled_lines = random.sample(all_lines, 128)

with open("guba_random_128.txt", "w", encoding="utf-8") as out_f:
    for line in sampled_lines:
        line = line.rstrip('\n')
        out_f.write(f"{line}\t<sep>\tlabel\n")
