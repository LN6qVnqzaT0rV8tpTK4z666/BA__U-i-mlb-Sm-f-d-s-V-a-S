.PHONY: build clean init-local init-docker

clean:
	bash scripts/clean__all.sh

build:
	@echo "==> Activate python virtual environment..."
	@bash -c "source .venv/bin/activate && \
		echo '==> Restarting services...' && \
		sh scripts/service__restart_all.sh && \
		echo '==> Starting EDNN training...' && \
		cd BA__Programmierung/ml/ && python3 ednn_regression__iris.py && \
		echo '==> Starting EDNN loss-visualization...' && \
		cd ../viz/ && python3 viz__ednn_regression__iris.py"

init all 
	$(MAKE) init-local
	$(MAKE) init-docker

init-docker:
	#@echo "==> Test Docker daemon"
	#docker --version

	sleep 3

	@echo "==> Starte Docker Desktop unter Windows"
	@cmd.exe /c start "" "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"

	sleep 3

	@echo "==> Build Docker image"
	docker build -t ba__projekt .

	@echo "==> Run Docker container"
	docker run --rm ba__projekt:latest

init-local:
	@echo "==> Set project path env variable"
	sh scripts/setup__project_base_path.sh

	@echo "==> Reload .bashrc"
	bash -c "source ~/.bashrc"

	@echo "==> Ensure required system packages are installed"
	sudo apt-get update && sudo apt-get install -y python3 tar unzip make

	@echo "==> Create virtual environment"
	python3 -m venv .venv

	@echo "==> Activate and install required Python packages"
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install duckdb matplotlib notebook numpy pandas torch scikit-learn scipy tensorflow torchvision

	@echo "==> Install local Python package"
	. .venv/bin/activate && pip install .

	@echo "==> Run project main entry point"
	. .venv/bin/activate && ba-programmierung

	bash scripts/clean__egg_info.sh