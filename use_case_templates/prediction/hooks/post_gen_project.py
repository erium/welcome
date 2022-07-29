import os
import shutil
import halerium_utilities as hu

# Select the specified template
model_templates = '{{cookiecutter.use_case_slug}}'
all_templates = ['regression', 'prediction_overview']
for template in all_templates:
    if template != model_templates:
        shutil.rmtree('./' + template)

hu.file.assign_new_card_ids_to_tree('./')