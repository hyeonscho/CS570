# import os
# import re

# # Specify the directory containing the files
# directory = "."  # Replace with the actual directory path

# # List all files in the directory
# for filename in os.listdir(directory):
#     # Match files with the specified pattern
#     match = re.match(r"(maze2d-giant-v1-linear)-round_(\d+)-(non_overlap-postprocess\.pkl)", filename)
#     if match:
#         base, num, rest = match.groups()
#         # Construct the new filename
#         new_filename = f"{base}-{rest}-round_{num}.pkl"
#         # Rename the file
#         old_path = os.path.join(directory, filename)
#         new_path = os.path.join(directory, new_filename)
#         os.rename(old_path, new_path)
#         print(f"Renamed: {filename} -> {new_filename}")

# print("Renaming complete.")


import os
import re

# Specify the directory containing the files
directory = "."  # Replace with the actual directory path

# List all files in the directory
for filename in os.listdir(directory):
    # Match files with the specified pattern
    names = filename.split("_")
    names = [n.replace("overlap", "overlap_harsh") for n in names]
    new_filename = "_".join(names)
    # Rename the file
    old_path = os.path.join(directory, filename)
    new_path = os.path.join(directory, new_filename)
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} -> {new_filename}")

print("Renaming complete.")
