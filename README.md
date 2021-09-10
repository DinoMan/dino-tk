# Dino Toolkit
Toolkit, containing modules and helper methods for machine learning projects. This package contains any modules and helper functions that I have used repeatedly in my projects. I hope it will prove useful to others as well.
The main package contains very general functionality as well as sub-packages tailored for specific operations (e.g. neural net modules, filesystem operations, etc.). These subpackages are outlined in the following sections.

## filesystem
The **filesystem** sub-package has functionality related to file operations. For example if you need to list all files in a directory you can do this with:
```
import dtk.filesystem as dfs
files = dfs.list_files("path")
```
If you have multiple directories with matching files (i.e. for storing corresponding data from different domains). You can find all the matching files by doing:
```
import dtk.filesystem as dfs
directories=["dir1", "dir2", "dir3"]
files = dfs.list_matching_files(directories)
```

## media
The **media** sub-package has functionality for handling media. With the methods in the media subpackage you can save media or convert to bytestream (for logging with [Weights&Biases](https://wandb.ai/site)).

## nn
The **nn** sub-package has useful building blocks that can be used to construct deep neural nets as well as losses. It builds on top of [Pytorch](https://pytorch.org/)

## transforms
The **transforms** sub-package contains useful transformations (mainly for sequential data)


