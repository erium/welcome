import os
import shutil
import halerium_utilities as hu
from halerium_utilities import notebook

# Select the specified template
model_templates = '{{cookiecutter.use_case_slug}}'
all_templates = ['regression', 'prediction_overview']
for template in all_templates:
    if template != model_templates:
        shutil.rmtree('./' + template)

hu.file.card_ids.assign_new_card_ids_to_tree('./')

# To move directory one up
folder = r"./"
subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

for sub in subfolders:
    for f in os.listdir(sub):
        src = os.path.join(sub, f)
        dst = os.path.join(folder, f)
        shutil.move(src, dst)
    shutil.rmtree(sub)

print("Hal_Magic_Template_done.")
