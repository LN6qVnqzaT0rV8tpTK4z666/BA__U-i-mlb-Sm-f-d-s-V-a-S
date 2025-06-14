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

Download your data-sources `assets/data/sources`.

Handle untar, unzip, .csv, move .compressed-src to same-named folder, move same-named folder from `assets/data/sources` to `assets/data/raw`.
```bash
make data-source
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

Build demo iris training
```bash
make build-training
```

Build whole project.
```bash
make build-main
```

Optional: Visualize loss-training with TensorBoard or inspect `viz`.
```bash
tensorboard --logdir /root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/data/processed/; http://localhost:6006
```

Optional: Clean build artifacts.
```bash
make clean
```

Optional: Build file-set by token(s), e.g.
```bash
bash scripts/print__folder_contents_by_depth.sh BA__Programmierung/viz 1 | \ 
bash scripts/create__files_by_token.sh test__ednn_regression py tests/
```

## Changelog
- Please check project gantt-chart from introduction mail.