# Architecture
City of Toronto - Recommendation System Architecture

## Development Notes  
Architecture, the development of the architecture design follows a trunk-based development pattern where features are developed in short-lived feature branches and continuously integrated back into the main branch. See more info about trunk-based workflow [here](https://trunkbaseddevelopment.com/). **The main branch will act as the production branch.**


### Local Development   
Download the software locally by running 
```console
git clone git@github.com:vatsal220/architecture.git
```
Then, navigate to the directory and create the python venv `architecture` environment through the following commands. Please ensure your Python version is 3.12 as that is the development version for this module.
```console
python3 -m venv architecture
source architecture/bin/activate
pip3 install -r requirements.txt
```

If you're using conda, the execute the following commands.
```console
conda create -n architecture python=3.11
conda activate architecture
```

## Environment Variables   
To run scripts in this repository, you require the necessary environment variables. These variables can either be in your `.bash_profile` or in a `.env` file created and stored in the base directory of this repository. The following holds the naming structure of the environment file necessary to execute scripts in this project.  

## Testing  
Execute unit tests from the main directory of this repository through the console using the following command:
```console   
make test-unit   
make test-integration
make test-e2e
```   