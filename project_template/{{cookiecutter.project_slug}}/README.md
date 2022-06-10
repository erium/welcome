{{cookiecutter.project_name}}
==============================

{{cookiecutter.description}}

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project
    │
    ├── applications       <- Code and scripts for setup and deployment of applications
    │   └──run.ipynb       <- Defines and runs a web-application
    │
    ├── data               <- Data dictionaries, manuals, and all other explanatory materials
    │   ├── 0_raw          <- The original, immutable data
    │   ├── 1_interim      <- Intermediate data that has been transformed
    │   ├── 2_prepared     <- The final data set, ready for preprocessing
    │   └── data.board     <- Describes the data model and structure
    │       
    ├── experiments        <- Notebooks and Boards structured via collaborator specific subfolders
    │   ├── {{ cookiecutter.author_slug }}              <- Subfolders for each author with timestamped file names/folders for each experiment
    │   ├── hypotheses_experiments_learnings.board      <- Structures the hypotheses-experiments-learnings process
    │   └── use_case_templates.ipynb                    <- Serves as entry point to Halerium data science use case templates
    │
    ├── knowledge 
    │   ├── references     <- Books, manuals, and all other explanatory materials
    │   ├── causal_graph.board      <- Board describing the causal dependencies in this use-case
    │   └── domain_knowledge.board  <- Board containing all domain and process specific knowledge
    │
    ├── output              <- Contains the assets generated in the course of the project
    │   ├── models          <- Trained and serialized models, model predictions, or model summaries       
    │   └── visualizations  <- Graphs and plots of data, models, and results
    │
    ├── pipelines 
    │   ├── pipeline1               <- Template folder for a pipeline utilizing mlflow     
    │   └── pipelines.board         <- Describes the individual pipelines, their functionalities and the relationships/process between them
    │
    ├── project_management
    │   ├── todos.board    <- Kanban Board to structure Todos
    │   ├── meeting_minutes<- Meeting minutes
    │   └── reports        <- Generated analyses as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so {{ cookiecutter.module_name }} can be imported
    ├── {{ cookiecutter.project_slug  }}.board        <- The high-level (excecutive) summary board
    └── {{ cookiecutter.module_name }}                <- Source code for use in this project.
        ├── __init__.py    <- Makes {{ cookiecutter.module_name }} a Python module
        │
        ├── module_understanding.py    <- Describes the high-level structure of  {{ cookiecutter.module_name }}
        │
        ├── application    <- A template Streamlit application
        │
        ├── data_io        <- Scripts to download or generate data
        │
        ├── features       <- Scripts to turn raw data into pre-processed data, i.e. features
        │
        ├── models         <- Scripts to build, train and evaluate models 
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations

