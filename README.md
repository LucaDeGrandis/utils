# utils
Various utils scripts.

# Installation
There is no installation required. However a "requirements.txt" file will be added in the future specifying critical pacakages and versions.
In order to import the scripts you can run the following snippets:

```
!git clone https://github.com/LucaDeGrandis/utils.git
!cd utils && pip install -r requirements_colab.txt

import os
import glob
import importlib.util
folder_path = "/content/utils/scripts"
py_files = glob.glob(os.path.join(folder_path, "*.py"))
for py_file in py_files:
    module_name = os.path.splitext(os.path.basename(py_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update({name: getattr(module, name) for name in dir(module) if callable(getattr(module, name))})
```

# Scripts details
Scripts details are available in each script
