.PHONY: clean build

clean:
	bash scripts/clean__all.sh

build:
	@echo "==> Restarting services..."
	sh scripts/service__restart_all.sh
	@echo "==> Starting EDNN training..."
	cd BA__Programmierung/ml/ && python3 ednn_regression__iris.py
	@echo "==> Starting EDNN loss-visualization..."
	cd BA__Programmierung/viz/ && python3 viz__ednn_regression__iris.py
