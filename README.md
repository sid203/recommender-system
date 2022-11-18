# Recommender-system

### Set up

Download `rating.csv` & `movie.csv` files from [kaggle link](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv)

Download also the trained model pickle files at this [google drive link](https://drive.google.com/file/d/1kD8-d98yrLy2BbmyQpzhMeW21bS64750/view?usp=sharing)
Please unzip these files and put it under `app/resources/` folder. 

Environment variables to set 
```bash
export ROOT_FOLDER=""
export PYTHONPATH="$PWD"

```

Install requirements 

`pip install -r requirements.txt`

Setup Git hooks

`pre-commit install #set up git hooks`

### Keep requirements clean and updated
Please make sure pip-tools is properly installed in your virtual environment with

```bash
pip install pip-tools

```
This should be done by adding the required package in the requirements.in file. 
Then using pip-tools the requirements.txt file can be reproduced. 

```
pip-compile --output-file requirements.txt --quiet requirements.in
pip-sync requirements.txt
```
Or using the makefile

```bash
make requirements
```

### Build docker image and run container
```bash
docker build -t registry.heroku.com/pagerank-webapp/web .
docker run --rm --name pagerank.webapp.container -e PORT=8080 -p 8080:8080 registry.heroku.com/pagerank-webapp/web:latest
```
Or using makefile 
```makefile
make dockerbuild
make dockerrun
```

To mount the datafiles in local, `cd` into project directory, uncomment the first 2 lines in `.dockerignore` and run:
```bash
make dockerrun-local-mount
```

### Deploy on Heroku 

Install heroku 

```bash
brew tap heroku/brew && brew install heroku
```

Login
```bash
heroku login #opens web browser
heroku container:login #to tell heroku we use container login for deployment
```

Choose a name for webapp on Heroku website after login:
`pagerank-webapp`

Push docker image & deploy
```bash
registry.heroku.com/pagerank-webapp/web
heroku container:release -a pagerank-webapp web
```

***Remarks***
The name of the docker image should be in this format:`registry.heroku.com/<app-name>/web`

### References
Pagerank modeling 

1. https://nlp.stanford.edu/IR-book/html/htmledition/topic-specific-pagerank-1.html

Deploy using Heroku 
1. https://testdriven.io/blog/fastapi-machine-learning/
