{{cookiecutter.project_name}}
==============================

{{cookiecutter.description}}

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Data dictionaries, manuals, and all other explanatory materials.
    │   ├── 0_raw          <- The original, immutable data dump.
    │   ├── 1_interim      <- Intermediate data that has been transformed.
    │   ├── 2_prepared     <- The final data set, ready for preprocessing
    │   ├── scripts        <- Script to load and transform the data (the initial ETL pipeline).
    │   └── data.board     <- Board describing the ETL process.
    │       
    ├── experiments        <- Notebooks and Boards structured via collaborator specific subfolders
    │   └── hypotheses_experiments_learnings.board      <- Board to structure the HET process
    │
    ├── knowledge 
    │   ├── references     <- Books, manuals, and all other explanatory materials.       
    │   ├── causal_graph.board      <- Board describing the causal dependencies in this use-case. 
    │   └── domain_knowledge.board  <- Board containing all domain and process specific knowledge. 
    │
    ├── project_management
    │   ├── todos.board    <- Kanban Board to structure Todos
    │   ├── meeting_minutes<- Meeting minutes
    │   └── reports        <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── run.ipynb          <-  with commands to, e.g., start the application
    ├── setup.py           <- makes project pip installable (pip install -e .) so {{ cookiecutter.module_name }} can be imported
    ├── {{ cookiecutter.project_slug  }}.board <- The high-level (excecutive) summary board
    └── {{ cookiecutter.module_name }}                <- Source code for use in this project.
        ├── __init__.py    <- Makes {{ cookiecutter.module_name }} a Python module
        │
        ├── module_understandig.py    <- Describes the high-level structure of  {{ cookiecutter.module_name }}.
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

