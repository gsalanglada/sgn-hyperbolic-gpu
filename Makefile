# =============================================================================
# SGN-waves-GPU -- build and test
# =============================================================================
#
# On MareNostrum 5 first:   source setup_env.src   (loads CUDA and OpenMPI)
# Build GPU/MPI targets from a login node that has the cuda module (alogin), and
# never rebuild while jobs that use build/ are queued: a queued job runs
# whatever binary is in build/ when it starts.
#
# TESTS (see the header of each file in test/ for what it checks)
#   make run-test      build + run the 12 HOST tests (any g++, no MPI, no GPU)
#   make run-mpi       CPU MPI vs serial, bitwise (mpirun, 1 / 4 / MPIRANKS ranks)
#   make run-cuda      GPU vs CPU parity, 1D and 2D (nvcc + a GPU node)
#   make run-cuda-mpi  multi-GPU vs single-GPU (2 and 4 ranks, one GPU each)
#
# PRODUCTION DRIVERS (multi-GPU; run with srun, see scripts/*.sbatch)
#   make bench         build/sgn_bench            scaling benchmark
#   make shoalfocus    build/shoal_focus_mpi      shoal focusing + geometry sweep
#   make tsunami       build/tsunami_margin_mpi   basin-scale tsunami
#
# VALIDATION DRIVERS (host; these targets also RUN the program into results/)
#   make beji          Beji-Battjes submerged bar
#   make bolles        random sea over an abrupt step
#   make alpha         relaxation-parameter study
#   make valfigs       solitary / collision / well-balancing figure data
#   make rkcompare     RK4 vs adaptive BS3(2)
#   make convergence   1D + 2D spatial order, temporal order
#   make ncptradeoff   central vs upwind NCP state, and the order-loss bisection
#   make showcase      single-GPU shoal animation (needs nvcc)
#   (single binaries: make build/convergence1d, make build/mms_bisect, ...)
#
#   make clean         remove build/
# =============================================================================

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
NVCC      ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2 -arch=sm_90    # MN5 ACC: H100 -> sm_90
MPICXX    ?= mpicxx
MPIRANKS  ?= 3
BUILD    := build

# Source groups linked into the drivers:
#   CORE_SRC     1D FR operators
#   SOLVER_SRC   integrators + 1D solver
#   SOLVER2_SRC  integrators + 1D + 2D solvers (2D drivers need both)
CORE_SRC   := src/core/fr_operators.cpp
SOLVER_SRC := src/solver/rk.cpp src/solver/sgn1d.cpp
SOLVER2_SRC := src/solver/rk.cpp src/solver/sgn1d.cpp src/solver/sgn2d.cpp

# Header dependencies. The physics and the FR operators are header-only, so a
# change there must rebuild everything that includes them. Listing only the .cpp
# files makes `make` report "Nothing to be done" after a physics edit and run a
# stale binary -- including the very gates meant to catch CPU/GPU drift.
PHYS_H := src/physics/hd.h src/physics/sgn_flux.h src/physics/sgn_flux2d.h \
          src/physics/sgn_dispersion.h
CORE_H := src/core/fr_operators.h src/core/mesh2d.h
SOLV_H := src/solver/rk.h src/solver/sgn1d.h src/solver/sgn2d.h
PAR_H  := src/parallel/slab1d.h src/parallel/sgn1d_mpi.h src/parallel/sgn2d_mpi.h
CUDA_H := src/cuda/sgn1d_cuda.h src/cuda/sgn2d_cuda.h src/cuda/sgn2d_cuda_mpi.h
DEPS_H := $(PHYS_H) $(CORE_H) $(SOLV_H)

# The 12 host tests: `make` (= all) builds them, `make run-test` builds and runs them.
TESTS := test_sgn_flux test_fr_operators test_rk test_sgn1d test_dispersion \
         test_solitary test_collision test_bathymetry test_slab1d \
         test_sgn_flux2d test_sgn2d test_relaxation

all: $(addprefix $(BUILD)/,$(TESTS))

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_sgn_flux: test/test_sgn_flux.cpp src/physics/sgn_flux.h src/physics/hd.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/test_fr_operators: test/test_fr_operators.cpp $(CORE_SRC) src/core/fr_operators.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(CORE_SRC) -o $@

$(BUILD)/test_rk: test/test_rk.cpp src/solver/rk.cpp src/solver/rk.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< src/solver/rk.cpp -o $@

$(BUILD)/test_sgn1d: test/test_sgn1d.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h src/physics/sgn_flux.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(SOLVER_SRC) $(CORE_SRC) -o $@

$(BUILD)/test_dispersion: test/test_dispersion.cpp $(SOLVER_SRC) $(CORE_SRC) src/physics/sgn_dispersion.h src/solver/sgn1d.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(SOLVER_SRC) $(CORE_SRC) -o $@

$(BUILD)/test_solitary: test/test_solitary.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h src/physics/sgn_flux.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(SOLVER_SRC) $(CORE_SRC) -o $@

$(BUILD)/test_collision: test/test_collision.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h src/physics/sgn_flux.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(SOLVER_SRC) $(CORE_SRC) -o $@

$(BUILD)/test_bathymetry: test/test_bathymetry.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h src/physics/sgn_flux.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(SOLVER_SRC) $(CORE_SRC) -o $@

$(BUILD)/test_slab1d: test/test_slab1d.cpp src/parallel/slab1d.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/test_sgn_flux2d: test/test_sgn_flux2d.cpp src/physics/sgn_flux2d.h src/physics/sgn_flux.h src/physics/hd.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/test_sgn2d: test/test_sgn2d.cpp $(SOLVER2_SRC) $(CORE_SRC) src/solver/sgn2d.h src/solver/sgn1d.h src/physics/sgn_flux2d.h src/core/mesh2d.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(SOLVER2_SRC) $(CORE_SRC) -o $@

$(BUILD)/test_relaxation: test/test_relaxation.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h src/physics/sgn_flux.h src/physics/sgn_dispersion.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(SOLVER_SRC) $(CORE_SRC) -o $@

run-test: all
	@set -e; for t in $(TESTS); do echo "== $$t =="; ./$(BUILD)/$$t; echo; done

# --- CUDA (GPU node: source setup_env.src first) ----------------------------
$(BUILD)/test_cuda_sgn1d: test/test_cuda_sgn1d.cu src/cuda/sgn1d_cuda.cu src/cuda/sgn1d_cuda.h $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h src/physics/sgn_flux.h $(DEPS_H) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) test/test_cuda_sgn1d.cu src/cuda/sgn1d_cuda.cu $(SOLVER_SRC) $(CORE_SRC) -o $@

$(BUILD)/test_cuda_sgn2d: test/test_cuda_sgn2d.cu src/cuda/sgn2d_cuda.cu src/cuda/sgn2d_cuda.h $(SOLVER2_SRC) $(CORE_SRC) $(DEPS_H) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) test/test_cuda_sgn2d.cu src/cuda/sgn2d_cuda.cu $(SOLVER2_SRC) $(CORE_SRC) -o $@

run-cuda: $(BUILD)/test_cuda_sgn1d $(BUILD)/test_cuda_sgn2d
	./$(BUILD)/test_cuda_sgn1d
	./$(BUILD)/test_cuda_sgn2d

# --- multi-GPU (CUDA-aware MPI; one GPU per rank) ----------------------------
$(BUILD)/test_cuda_mpi_sgn2d: test/test_cuda_mpi_sgn2d.cu src/cuda/sgn2d_cuda_mpi.cu src/cuda/sgn2d_cuda.cu src/parallel/sgn2d_mpi.cpp $(SOLVER2_SRC) $(CORE_SRC) src/cuda/sgn2d_cuda_mpi.h $(DEPS_H) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -ccbin=$(MPICXX) test/test_cuda_mpi_sgn2d.cu src/cuda/sgn2d_cuda_mpi.cu src/cuda/sgn2d_cuda.cu src/parallel/sgn2d_mpi.cpp $(SOLVER2_SRC) $(CORE_SRC) -o $@

run-cuda-mpi: $(BUILD)/test_cuda_mpi_sgn2d
	mpirun -n 2 ./$(BUILD)/test_cuda_mpi_sgn2d
	mpirun -n 4 ./$(BUILD)/test_cuda_mpi_sgn2d

# --- scaling benchmark (see scripts/bench.sbatch) ----------------------------
$(BUILD)/sgn_bench: apps/sgn_bench.cu src/cuda/sgn2d_cuda_mpi.cu src/parallel/sgn2d_mpi.cpp $(SOLVER2_SRC) $(CORE_SRC) src/cuda/sgn2d_cuda_mpi.h $(DEPS_H) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -ccbin=$(MPICXX) apps/sgn_bench.cu src/cuda/sgn2d_cuda_mpi.cu src/parallel/sgn2d_mpi.cpp $(SOLVER2_SRC) $(CORE_SRC) -o $@

bench: $(BUILD)/sgn_bench

# --- 2D production case: shoal focusing on the distributed solver ------------
$(BUILD)/shoal_focus_mpi: apps/shoal_focus_mpi.cu src/cuda/sgn2d_cuda_mpi.cu src/parallel/sgn2d_mpi.cpp $(SOLVER2_SRC) $(CORE_SRC) src/cuda/sgn2d_cuda_mpi.h $(DEPS_H) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -ccbin=$(MPICXX) apps/shoal_focus_mpi.cu src/cuda/sgn2d_cuda_mpi.cu src/parallel/sgn2d_mpi.cpp $(SOLVER2_SRC) $(CORE_SRC) -o $@

shoalfocus: $(BUILD)/shoal_focus_mpi

# --- capability case: dispersive tsunami over a continental margin -----------
$(BUILD)/tsunami_margin_mpi: apps/tsunami_margin_mpi.cu src/cuda/sgn2d_cuda_mpi.cu src/parallel/sgn2d_mpi.cpp $(SOLVER2_SRC) $(CORE_SRC) src/cuda/sgn2d_cuda_mpi.h src/io/vtk2d_mpi.h $(DEPS_H) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -ccbin=$(MPICXX) apps/tsunami_margin_mpi.cu src/cuda/sgn2d_cuda_mpi.cu src/parallel/sgn2d_mpi.cpp $(SOLVER2_SRC) $(CORE_SRC) -o $@

tsunami: $(BUILD)/tsunami_margin_mpi

# --- validation cases (host) -------------------------------------------------
$(BUILD)/beji_battjes: apps/beji_battjes.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h src/physics/sgn_dispersion.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) apps/beji_battjes.cpp $(SOLVER_SRC) $(CORE_SRC) -o $@

beji: $(BUILD)/beji_battjes
	mkdir -p results
	./$(BUILD)/beji_battjes

$(BUILD)/bolles_step: apps/bolles_step.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h src/physics/sgn_dispersion.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) apps/bolles_step.cpp $(SOLVER_SRC) $(CORE_SRC) -o $@

bolles: $(BUILD)/bolles_step
	mkdir -p results
	./$(BUILD)/bolles_step

$(BUILD)/alpha_sweep: apps/alpha_sweep.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) apps/alpha_sweep.cpp $(SOLVER_SRC) $(CORE_SRC) -o $@

alpha: $(BUILD)/alpha_sweep
	mkdir -p results
	./$(BUILD)/alpha_sweep

$(BUILD)/val_figures: apps/val_figures.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) apps/val_figures.cpp $(SOLVER_SRC) $(CORE_SRC) -o $@

valfigs: $(BUILD)/val_figures
	mkdir -p results
	./$(BUILD)/val_figures

$(BUILD)/rk_compare: apps/rk_compare.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) apps/rk_compare.cpp $(SOLVER_SRC) $(CORE_SRC) -o $@

rkcompare: $(BUILD)/rk_compare
	mkdir -p results
	./$(BUILD)/rk_compare

$(BUILD)/convergence1d: apps/convergence1d.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) apps/convergence1d.cpp $(SOLVER_SRC) $(CORE_SRC) -o $@

$(BUILD)/convergence2d: apps/convergence2d.cpp $(SOLVER2_SRC) $(CORE_SRC) src/solver/sgn2d.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) apps/convergence2d.cpp $(SOLVER2_SRC) $(CORE_SRC) -o $@

$(BUILD)/convergence_time: apps/convergence_time.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) apps/convergence_time.cpp $(SOLVER_SRC) $(CORE_SRC) -o $@

$(BUILD)/ncp_tradeoff: apps/ncp_tradeoff.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) apps/ncp_tradeoff.cpp $(SOLVER_SRC) $(CORE_SRC) -o $@

$(BUILD)/mms_bisect: apps/mms_bisect.cpp $(SOLVER_SRC) $(CORE_SRC) src/solver/sgn1d.h $(DEPS_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) apps/mms_bisect.cpp $(SOLVER_SRC) $(CORE_SRC) -o $@

ncptradeoff: $(BUILD)/ncp_tradeoff $(BUILD)/mms_bisect
	mkdir -p results
	@echo "=== central vs upwind interface state ==="; ./$(BUILD)/ncp_tradeoff
	@echo; echo "=== bisecting the origin of the parity loss ==="; ./$(BUILD)/mms_bisect

convergence: $(BUILD)/convergence1d $(BUILD)/convergence2d $(BUILD)/convergence_time
	mkdir -p results
	@echo "=== 1D convergence (MMS, spatial) ==="; ./$(BUILD)/convergence1d
	@echo "=== 2D convergence (MMS, spatial) ==="; ./$(BUILD)/convergence2d
	@echo "=== temporal convergence (BS3 vs RK4) ==="; ./$(BUILD)/convergence_time

# --- 2D showcase (single GPU + VTK frames for ParaView) ----------------------
$(BUILD)/showcase_shoal: apps/showcase_shoal.cu src/cuda/sgn2d_cuda.cu $(SOLVER2_SRC) $(CORE_SRC) src/io/vtk2d.h $(DEPS_H) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) apps/showcase_shoal.cu src/cuda/sgn2d_cuda.cu $(SOLVER2_SRC) $(CORE_SRC) -o $@

showcase: $(BUILD)/showcase_shoal
	mkdir -p results
	./$(BUILD)/showcase_shoal

# --- MPI (CPU multi-rank reproducibility oracle) ----------------------------
$(BUILD)/test_mpi_sgn1d: test/test_mpi_sgn1d.cpp src/parallel/sgn1d_mpi.cpp $(SOLVER_SRC) $(CORE_SRC) src/parallel/sgn1d_mpi.h src/parallel/slab1d.h $(DEPS_H) | $(BUILD)
	$(MPICXX) $(CXXFLAGS) test/test_mpi_sgn1d.cpp src/parallel/sgn1d_mpi.cpp $(SOLVER_SRC) $(CORE_SRC) -o $@

$(BUILD)/test_mpi_sgn2d: test/test_mpi_sgn2d.cpp src/parallel/sgn2d_mpi.cpp $(SOLVER2_SRC) $(CORE_SRC) src/parallel/sgn2d_mpi.h src/parallel/slab1d.h $(DEPS_H) | $(BUILD)
	$(MPICXX) $(CXXFLAGS) test/test_mpi_sgn2d.cpp src/parallel/sgn2d_mpi.cpp $(SOLVER2_SRC) $(CORE_SRC) -o $@

run-mpi: $(BUILD)/test_mpi_sgn1d $(BUILD)/test_mpi_sgn2d
	mpirun -n $(MPIRANKS) ./$(BUILD)/test_mpi_sgn1d
	@echo "--- 1D: also size=1 (bitwise-serial) and size=4 ---"
	mpirun -n 1 ./$(BUILD)/test_mpi_sgn1d
	mpirun -n 4 ./$(BUILD)/test_mpi_sgn1d
	@echo "--- 2D ---"
	mpirun -n $(MPIRANKS) ./$(BUILD)/test_mpi_sgn2d
	mpirun -n 1 ./$(BUILD)/test_mpi_sgn2d
	mpirun -n 4 ./$(BUILD)/test_mpi_sgn2d

clean:
	rm -rf $(BUILD)

.PHONY: all run-test run-cuda run-mpi run-cuda-mpi bench shoalfocus beji bolles alpha valfigs rkcompare convergence ncptradeoff showcase clean
