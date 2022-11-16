.PHONY: requirements
requirements:
	pip-compile --output-file requirements.txt --quiet requirements.in
	pip-sync requirements.txt
.PHONY: typehint
typehint:
	mypy --config-file pyproject.toml app/
.PHONY: clean
clean:
	find . -type d -name .mypy_cache | xargs rm -fr
.PHONY: black
black:
	black app/ tests/
.PHONY: dockerbuild
dockerbuild:
	docker build -t registry.heroku.com/pagerank-webapp/web .
.PHONY: dockerrun-local-mount
dockerrun-local-mount:
	docker run --rm -v ${CURDIR}/app/resources/:/recommender-system/app/resources/ --name recommender.container.light pagerank-light:$(version) --userid=$(user)
.PHONY: dockerrun
dockerrun:
	docker run --rm --name pagerank.webapp.container -e PORT=8080 -p 8080:8080 registry.heroku.com/pagerank-webapp/web:latest