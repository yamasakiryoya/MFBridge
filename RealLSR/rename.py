import os
import glob

file_names = glob.glob('**/.DS_Store', recursive=True)
print(file_names)
for file_name in file_names: os.remove(file_name)

file_names = glob.glob('**/._.DS_Store', recursive=True)
print(file_names)
for file_name in file_names: os.remove(file_name)

file_names = glob.glob('**/*.pt', recursive=True)
print(file_names)
for file_name in file_names: os.remove(file_name)