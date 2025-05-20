.PHONY: download train mutate deploy check lint test

PYTHON=python

download:
	$(PYTHON) -m elastic_trader.data.download_data > /dev/null

train:
	$(PYTHON) -m elastic_trader.scripts.train

mutate:
	$(PYTHON) -m elastic_trader.scripts.mutate $(N)

deploy:
	$(PYTHON) -m elastic_trader.scripts.deploy --date $(DATE)

lint:
	pylint elastic_trader || true
	mypy elastic_trader || true

test:
	pytest -q

check: lint test
