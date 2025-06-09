# BA: U-i-mlb-Sm-f-d-s-V-a-S 

Bachelor-Arbeit: Unsicherheiten in machine-learning-basierten Surrogatmodellen für die szenariobasierte Validierung autonomer Systeme

## TOC

- [Initialization](#initialization)
- [Structure](#structure)
- [Changelog](#changelog)

## Initialization

Setup with Windows 11 WSL-2-Ubuntu-24.04.01-LTS.

Ensure installation of python3, tar, unzip, make.
```bash
sudo apt install python3 tar unzip make
```

Setup project in local python package and in docker image.
```bash
make init-all
```

Setup project in local python package.
```bash
make init-local
```

Setup project in docker.
```bash
make init-docker
```

Demo iris training
```bash
make build
```

Optional: Visualize loss-training with TensorBoard or inspect `viz`.
```bash
tensorboard --logdir /root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/data/processed/; http://localhost:6006
```

Optional: Clean build artifacts.
```bash
make clean
```

## Changelog
- [x] Setup basic make build, clean, init-all, init-local, init-docker.
- [x] Start english documentation.
- [x] Setup basic loss-visualization to `/viz/`.
- [x] Setup basic iris EDNN-training to `/data/processed/`.
- [x] Setup basic Docker.
- [x] Setup Readme.