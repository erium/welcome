import os
import shutil
import halerium_utilities as hu
from halerium_utilities import notebook

# Select the specified template
model_templates = '{{cookiecutter.use_case_slug}}'
all_templates = ["bayesian_optimization",
                 "classical_doe", "bayesian_modelling", "doe_overview"]
for template in all_templates:
    if template != model_templates:
        shutil.rmtree('./' + template)

hu.file.card_ids.assign_new_card_ids_to_tree('./')

print("Hal_Magic_Template_done.")
