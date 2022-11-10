# recommender-system

Install requirements 

`pip install -r requirements.txt`

Setup Git hooks

`pre-commit install #set up git hooks`

### Keep requirements clean and updated

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
