# Interpolation Methods
In this section, we explore some possible methods with/without AI to interpolate frames

More specifically, we leverage existing methods such as RIFE to interpolate frames

## Requirements
We need to install FILM. To do that, we recommend to leverage the existing docker container image:

Download/run container
```
docker run -it --rm   --gpus all   -v $(pwd):/workspace   -w /workspace   --name tf_film   gcr.io/deeplearning-platform-release/tf2-gpu.2-6:latest   bash
```

## Install dependencies inside the container after running it
```
pip install opencv-python
apt update
apt-get install -y libgl1-mesa-glx
apt install ffmpeg -y
```

### Known issues

1. E: Unable to locate package nvidia-docker2

Solution:

```
distribution="ubuntu20.04"  # or ubuntu22.04 if that's your system

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker-archive-keyring.gpg

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-docker-archive-keyring.gpg] https://#' | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list > /dev/null

sudo apt update
sudo apt install -y nvidia-docker2

sudo systemctl restart docker
```
