# NeRF-Glasses

This repository provides the implementation of our hybrid renderer combining neural rendering (Instant NGP) with traditional rendering (surface meshes).
> __[View-Consistent Virtual Try-on of Glasses using a Hybrid NeRF-Mesh Rendering Approach](https://diglib.eg.org/items/e5a78d0d-e3df-4dfc-a423-27549064f2ff)__  
> [Arne Rak](https://orcid.org/0000-0001-6385-3455), [Tristan Wirth](https://orcid.org/0000-0002-2445-9081),  [Thomas Lindemeier](https://orcid.org/0009-0003-7715-8439), [Volker Knauthe](https://orcid.org/0000-0001-6993-5099), [Arjan Kuijper](https://orcid.org/0000-0002-6413-0061)  
> _Computer Graphics & Visual Computing (CGVC) 2024_  

A Dockerfile is provided for ease of use.

## Prerequisites

1. NVIDIA GPU (RTX 20 Series and above)
2. CUDA Toolkit (11.2+)
3. [Docker and Docker Compose](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

## Capturing a Dataset
For best results, please adhere to the following guidelines.

- Capture at least 180° views of the subject
- The subject should not move their head or vary their facial expression
- Frames in which the subject is blinking should be removed
- The final dataset should consist of jpeg/png images or an mp4 video

An example dataset (jpeg images) and a trained NeRF model is provided in the datasets folder.

## Docker Container
```
# build image (this takes about 30 minutes)
docker compose -f ./docker-compose.yml build nerf-glasses
# do xhost magic
xhost local:root
# open bash inside the container
docker compose -f ./docker-compose.yml run nerf-glasses /bin/bash
```

## Training NeRF Model
Colmap and Instant-NGP (iNGP) are used for training the NeRF model (.msgpack) using a given dataset. All dependencies are included in the Docker container.

1. Place images or video in the ./datasets folder, so they can be accessed from inside the container
2. Open bash inside the iberstruct Docker container
   ```
   # open bash inside the container
   docker compose -f ./docker-compose.yml run iberstruct /bin/bash
   # cd into the desired dataset folder
   cd ./datasets/my_dataset
   ```
3. Run iNGP’s colmap2nerf script to process the dataset. This will produce a transforms.json file that can be used by iNGP and other NeRF implementations.
   1. For images (use `--colmap_matcher exhaustive` for non-sequential images):
      ```
      python3 /opt/instant-ngp/scripts/colmap2nerf.py --colmap_matcher sequential --run_colmap --images ./images
      ```
   2. For videos:
      ```
      python3 /opt/instant-ngp/scripts/colmap2nerf.py --video_in video.mp4 --video_fps 5 --run_colmap
      ```
4. Run train.py on the dataset. This will generate a `nerf.msgpack` file
    ```
    cd /volume
    python3 train.py ./datasets/my_dataset
    ```

## Placing Glasses on Heads
Our `nerf_mesh_renderer` application is used to render NeRFs and Meshes together in the same scene.
It provides necessary bindings for placing glasses on heads using only python:
- Loading NeRFs
- Loading meshes with a given translation, scale and rotation
- Getting & setting camera pose
- Rendering images
- Removing floaties

`render.py` is an example application for achieving this.
The involved steps are:
1. Load NeRF model specified in arguments
2. Rotate NeRF model such that the face is facing the camera, using MediaPipe and `reference_landmarks.npy`
3. Get 2D face landmarks from multiple views to then triangulate them to 3D landmarks
4. The glasses mesh specified in arguments is loaded and transformed, such that its base and temples align with the 3d landmarks

For `render.py` to function, the origin of the glasses mesh coordinate system must coincide with the glasses' base, i.e. the vertex that is supposed to sit on the nose must have coordinates (0, 0, 0).
The arguments for `render.py` are the filepaths to the NeRF and mesh, and two vertex positions of the glasses' temples.

Example usage:
```
cd /volume
python3 render.py -n ./datasets/my_dataset/nerf.msgpack -m /opt/nmr/assets/meshes/glasses/glasses.gltf -l "-0.732 -1.002 -0.057" -r "0.732 -1.002 -0.057"
```
