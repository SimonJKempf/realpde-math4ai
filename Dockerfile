# RealPDE 2026 Track 1 -- final training package (reproduces Codabench submission #20, sub38).
# Same base as the organizers' template: the torch the development-phase grader ran.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# h5py reads the raw release; everything else the code needs ships with the base image
# (torch 2.2.2, numpy 1.26) or inside /workspace (einops is vendored under kit/_vendor).
RUN pip install --no-cache-dir h5py==3.11.0

# PYTHONPATH: the package `pde_competition` lives in src/ and is imported by every script.
# TORCHDYNAMO_DISABLE: flower_standalone decorates two functions with @torch.compile; under
#   torch 2.2.2 that would drive the inductor backend, which needs a C++ toolchain the runtime
#   image does not have. Disabled, they run eagerly -- how every shipped model was trained.
ENV PYTHONPATH=/workspace/src \
    TORCHDYNAMO_DISABLE=1 \
    WANDB_MODE=disabled \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
COPY . /workspace
