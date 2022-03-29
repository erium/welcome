import os
import shutil
import halerium_utilities as hu

# Select the specified template
model_templates = '{{cookiecutter.model_templates}}'
all_templates = ['1-regression'] 
for template in all_templates:
    if template != model_templates:
        shutil.rmtree('{{ cookiecutter.use_case_folder }}/' + template)
        
hu.file.assign_new_card_ids_to_tree('./')

# Create new cards in project template folder
project_path = './../../hypotheses_experiments_learnings.board'
if os.path.exists(project_path):
    learning_card = hu.board.board.create_card(title='Learning',
                                                content='*Enter learning here*',
                                                position={"x":935.066650390625,"y":333},
                                                size={"width":100,"height":130},
                                                color="#125059")
    learning_card_id = learning_card['id']

    experiment_card = hu.board.board.create_card(title='Experiment',
                                                content=model_templates + ' outlier detection',
                                                position={"x":690.816650390625,"y":333},
                                                size={"width":100,"height":130},
                                                color="#28337e")
    experiment_card_id = experiment_card['id']

    hypo_ques_card = hu.board.board.create_card(title='Hypothesis/Question',
                                                content='*Enter hypothesis here*',
                                                position={"x":371.254150390625,"y":333},
                                                size={"width":173,"height":130},
                                                color="#513063")
    hypo_ques_card_id = hypo_ques_card['id']

    hu.board.board.add_card_to_board(project_path, learning_card)
    hu.board.board.add_card_to_board(project_path, experiment_card)
    hu.board.board.add_card_to_board(project_path, hypo_ques_card)