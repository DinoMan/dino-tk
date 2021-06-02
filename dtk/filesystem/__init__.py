import os
from .filtering import Filter, PatternExtractor, Tagger
import re
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from functools import reduce
import random
import arrow


def remove_old_files(folder, ext=None, weeks=0, days=0, hours=0, suppress_warnings=True):
    files = list_files(folder, file_filter=Filter(ext=ext))
    critical_time = arrow.now().shift(weeks=-weeks).shift(days=-days).shift(hours=-hours)
    for f in files:
        try:
            file_time = arrow.get(os.stat(f).st_mtime)
            if file_time < critical_time:
                os.remove(f)
        except Exception as e:
            if suppress_warnings:
                pass
            else:
                warnings.warn(str(e), RuntimeWarning)


def get_common_letters(strlist):
    return ''.join([x[0] for x in zip(*strlist) if reduce(lambda a, b: (a == b) and a or None, x)])


def find_common_start(file_list):
    prev = None
    while True:
        common = get_common_letters(file_list)
        if common == prev:
            break
        file_list.append(common)
        prev = common

    return os.path.dirname(get_common_letters(file_list))


class ImageDataset(Dataset):
    def __init__(self, folder, resize=(256, 256), random_flip=0, ext=None):
        self.files = list_files(folder, file_filter=Filter(ext=ext))
        self.image_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=random_flip),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, sample_idx):
        return self.image_transform(Image.open(self.files[sample_idx]).convert("RGB"))

    def __len__(self):
        return len(self.files)


def filify(string):
    filename = string.replace(" ", "_")
    filename = filename.replace(":", "-")
    filename = filename.replace("-_", "-")
    return filename


def find_extensions(file_name, allowed_exts=None):
    if allowed_exts is not None:
        if not isinstance(allowed_exts, list):
            allowed_exts = [allowed_exts]

        ext_list = []
        for ext in allowed_exts:
            if os.path.isfile(file_name + ext):
                ext_list.append(ext)

        return ext_list

    path = os.path.dirname(file_name)
    name = os.path.basename(file_name)

    files = os.listdir(path)
    rx = r'({0}\.\w+)'.format(name)

    matched_files = re.findall(rx, " ".join(files))
    return [os.path.splitext(m)[1] for m in matched_files]


def list_files(folder, file_filter=None):
    file_list = []
    for root, dirs, files in os.walk(folder, topdown=False):
        if not dirs and files:
            for f in files:
                file_path = os.path.join(root, f)
                if file_filter is None or file_filter(file_path):
                    file_list.append(file_path)

    return file_list


def list_matching_files(directories, ext=None):
    file_filters = []
    for i in range(len(directories)):  # For every directory
        if ext is None or ext[i] is None:
            file_filters.append(Filter(is_file=True))  # Create a filter which just checks that the file exists
        else:
            file_filters.append(Filter(is_file=True, ext=ext[i]))  # Create a filter that requires extensions

    matching_files = []
    matching_dirs = []
    matching_exts = []
    for root, dirs, files in os.walk(directories[0], topdown=False):
        for name in files:
            file_name = os.path.splitext(name)[0]
            extensions = [os.path.splitext(name)[1]]
            if not file_filters[0](os.path.join(root, name)):
                continue

            sub_path = root.replace(directories[0], "")
            if sub_path != '' and sub_path[0] == '/':
                sub_path = sub_path[1:]

            # Now search for matching files in the other directories
            all_matches_found = True
            for i in range(1, len(directories)):

                # Find all the possible extensions of the file in the directory
                if ext is None:
                    possible_extensions = find_extensions(os.path.join(directories[i], sub_path, file_name))
                else:
                    possible_extensions = find_extensions(os.path.join(directories[i], sub_path, file_name), ext[i])

                if not possible_extensions:  # If you can't find any files
                    all_matches_found = False  # Report that not all matches were found and break
                    break

                # If we have found files with that name
                match_found = False
                for possible_extension in possible_extensions:  # Check all possible extensions to find only valid ones
                    if file_filters[i](os.path.join(directories[i], sub_path, file_name + possible_extension)):
                        match_found = True  # If we foind a suitable one then report and break
                        extensions.append(possible_extension)
                        break

                if not match_found:  # If no match was found report that not all matches were found
                    all_matches_found = False
                    break

            if not all_matches_found:
                continue

            matching_dirs.append(sub_path)
            matching_files.append(file_name)
            matching_exts.append(extensions)

    return {"files": matching_files, "dirs": matching_dirs, "exts": matching_exts}
