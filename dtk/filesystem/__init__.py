import os
from .filtering import Filter

def list_files(folder, file_filter=None):
    file_list = []
    for root, dirs, files in os.walk(folder, topdown=False):
        if not dirs and files:
            for f in files:
                file_path = os.path.join(root, f)
                if file_filter is None or file_filter(file_path):
                    file_list.append(file_path)

    return file_list
