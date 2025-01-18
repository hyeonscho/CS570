# FROM USERNAME/rlx:torch
FROM doojin/rlx:torch

# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update && apt-get install -y \
  freeglut3-dev \
  xvfb \
  xserver-xephyr \
  python-opengl \
  python3-opencv \
  && apt-get clean
  
# RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# RUN wget -q --show-progress\
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 
# RUN conda --version

RUN pip install torchsummary
RUN pip install plotly tensorboardX
# RUN pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
RUN pip install pudb einops
RUN pip install jupyterlab notebook scikit-learn
ENV PYTHONBREAKPOINT "pudb.set_trace"

# osmesa: modification?!
# RUN apt-get update && apt-get install -y libgl1-mesa-glx libosmesa6
# ENV DISABLE_MUJOCO_RENDERING 1
# ENV MUJOCO_GL osmesa

# egl: Original
# RUN apt-get update && apt-get install -y libglew2.1
# ENV MUJOCO_GL egl


# Copy and Directory
# COPY choreographer/conda_env.yml /tmp/conda_env.yml
# RUN conda env create -f /tmp/conda_env.yml

RUN pip install numpy==1.19.5 \
                absl-py==0.13.0 \
                pyparsing==2.4.7 \
                jupyterlab==4.3.4 \
                scikit-image==0.18.1 \
                termcolor==1.1.0 \
                dm-control==0.0.364896371 \
                tb-nightly==2.10.0a20220724 \
                imageio==2.9.0 \
                imageio-ffmpeg==0.4.4 \
                hydra-core==1.1.0 \
                hydra-submitit-launcher==1.1.5 \
                pandas==1.3.0 \
                ipdb==0.13.9 \
                yapf==0.31.0 \
                # mujoco_py==2.0.2.13 \
                sklearn==0.0 \
                matplotlib==3.4.2 \
                opencv-python \
                wandb==0.15.4 \
                moviepy==1.0.3 \
                git+https://github.com/rlworkgroup/metaworld.git@18118a28c06893da0f363786696cc792457b062b#egg=metaworld \
                pyglet==1.5.24 \
                imagehash==4.3.1 \
                hexhamming==2.2.3

# https://github.com/openai/mujoco-py/issues/773
RUN pip install cython==0.29.36
RUN pip install mujoco_py==2.0.2.13 --no-build-isolation

# https://github.com/pypa/setuptools/issues/3301
RUN pip install setuptools==65.5.0
RUN pip install scikit-image

# COPY rlx_envs/ /rlx_envs/
# RUN chown -R 1000:root /rlx_envs/ && chmod -R 775 /rlx_envs
# WORKDIR /rlx_envs
# RUN pip install --no-cache-dir -e .
# WORKDIR /rlx_envs/rlx_envs/memory-maze
# RUN pip install --no-cache-dir -e .

# RUN pip install minari \
                # git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl


# COPY vd4rl/ /vd4rl/
# RUN chown -R 1000:root /vd4rl/ && chmod -R 775 /vd4rl
# WORKDIR /vd4rl
# RUN pip install --no-cache-dir -e .

# RUN pip install tables==3.6.1


RUN python -m pip install setuptools==66 wheel==0.38.1

RUN pip install numpy \
            setuptools==65.0.0 \
            "wheel<=0.40.0" \
            gym==0.23.1 \
            cython==0.29.36 \
            mujoco-py==2.0.2.13 \
            matplotlib==3.3.4 \
            typed-argument-parser \
            git+https://github.com/Farama-Foundation/d4rl@f2a05c0d66722499bf8031b094d9af3aea7c372b#egg=d4rl \
            scikit-image==0.17.2 \
            scikit-video==1.1.11 \
            gitpython \
            einops \
            ffmpeg \
            ffprobe \
            pillow \
            tqdm \
            pandas \
            wandb \
            # flax >= 0.3.5 \
            # jax <= 0.2.21 \
            ray==2.0.0 \
            crcmod \
            google-api-python-client  \
            cryptography \
            git+https://github.com/JannerM/doodad.git@janner  \
            gdown>=4.6.0


RUN pip install ml-logger==0.8.69 params-proto==2.9.6

WORKDIR /

COPY scripts_docker/mjkey.txt /root/.mujoco/mjkey.txt
COPY scripts_docker/arial.ttf /tmp/arial.ttf
WORKDIR /root/


ENV MJLIB_PATH /.mujoco/mujoco200_linux/bin/libmujoco200.so
ENV MJKEY_PATH /.mujoco/mjkey.txt

ENV ML_LOGGER_ROOT http://0.0.0.0:8081
ENV ML_LOGGER_USER USERNAME



RUN chmod -R 777 /usr/local/lib/python3.8/dist-packages/mujoco_py/generated/

ARG USER_ID
ARG GROUP_ID

RUN groupadd -g $GROUP_ID usergroup && \
    useradd -l -u $USER_ID -g usergroup USERNAME && \
    install -d -m 0755 -o USERNAME -g usergroup /home/USERNAME

USER USERNAME

RUN git config --global --add safe.directory '*'
