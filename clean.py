import os
import shutil

# Directories to clean
directories = [".", "ai", "games"]

# File extensions to remove
extensions = [".c", ".so", ".html"]

# Remove files with specified extensions
for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(tuple(extensions)):
            os.remove(os.path.join(directory, filename))
            print(f"Removed: {os.path.join(directory, filename)}")

# Remove the build directory
if os.path.exists("build"):
    shutil.rmtree("build")
    print("Removed: build directory")
