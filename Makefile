CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
PYTHON ?= python3

release: BUILDTYPE=release
release: BUILD_DIR=build/release
debug: BUILDTYPE=debug
debug: BUILD_DIR=build/debug
cuda-release: BUILDTYPE=release
cuda-release: BUILD_DIR=build/cuda-release
cuda-release: CMAKE_EXTRA=-DGPG_USE_CUDA=ON $(if $(CUDA_ARCH),-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH),)
cuda-debug: BUILDTYPE=debug
cuda-debug: BUILD_DIR=build/cuda-debug
cuda-debug: CMAKE_EXTRA=-DGPG_USE_CUDA=ON $(if $(CUDA_ARCH),-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH),)
release: main-build
debug: main-build
cuda-release: main-build
cuda-debug: main-build

main-build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
	cmake -S ../../ -B . -DCMAKE_PREFIX_PATH=$(CONDA_PREFIX) -DCMAKE_BUILD_TYPE=$(BUILDTYPE) $(CMAKE_EXTRA) && \
	make && \
	cp -r ../../src/pypkg . && \
	mv _pb_gpg*.so pypkg/pygpg/_pb_gpg.so && \
	cd pypkg && \
	$(PYTHON) -m pip install --user --force-reinstall --no-deps . && \
	cd ../ && \
	rm -r pypkg


clean:
	rm -rf build
