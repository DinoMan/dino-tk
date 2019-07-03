import os
from .filtering import Filter, PatternExtractor, Tagger

def list_files(folder, file_filter=None):
    file_list = []
    for root, dirs, files in os.walk(folder, topdown=False):
        if not dirs and files:
            for f in files:
                file_path = os.path.join(root, f)
                if file_filter is None or file_filter(file_path):
                    file_list.append(file_path)

    return file_list


def list_matching_files(directories, ext=None, file_filter=None):
    if file_filter is None:
        file_filter = Filter(is_file=True)

    matching_files = []
    matching_dirs = []
    for root, dirs, files in os.walk(directories[0], topdown=False):
        for name in files:
            file_name = os.path.splitext(name)[0]
            extension = os.path.splitext(name)[1]

            if ext is not None and extension != ext[0]:
                continue

            sub_path = root.replace(directories[0], "")
            if sub_path != '' and sub_path[0] == '/':
                sub_path = sub_path[1:]

            criteria_pass = True
            for i in range(1, len(directories)):
                if not file_filter(os.path.join(directories[i], sub_path, file_name + ext[i])):
                    criteria_pass = False
                    break

            if not criteria_pass:
                continue

            matching_dirs.append(sub_path)
            matching_files.append(file_name)

    return {"files": matching_files, "dirs": matching_dirs}
