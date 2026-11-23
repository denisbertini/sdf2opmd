//
// Minimal stress test: exercises exactly the openPMD write pattern that
// sdf2opmd uses (file-based BP5, chunked store of unique_ptr buffers,
// attributes, batched flushes, iteration close) with multiple ranks,
// independently of any SDF code. Used to isolate openPMD/ADIOS2 issues.
//
// usage: stress_openpmd <outdir>
//
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>

#include <openPMD/openPMD.hpp>

int
main(int argc, char * * argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: stress_openpmd <outdir>\n");
        return 2;
    }
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    using namespace openPMD;
    bool const noParticles = getenv("NO_PARTICLES") != nullptr;
    std::string const path =
        std::string(argv[1]) + "/stress_%04T.bp";
    Series series(path, Access::CREATE, MPI_COMM_WORLD);
    series.setIterationEncoding(IterationEncoding::fileBased);

    for (uint64_t step = 1; step <= 2; ++step)
    {
        Iteration it = series.iterations[step];
        it.setTime(double(step) * 1.5e-15);
        it.setDt(5e-17);
        it.setAttribute("sdf_source", "stress");

        // 3D mesh, split along x across ranks, like readFieldSlab does
        Mesh mesh = it.meshes["E"];
        mesh.setGeometry(Mesh::Geometry::cartesian);
        mesh.setAxisLabels({"x", "y", "z"});
        mesh.setGridSpacing(std::vector<double>{0.1, 0.2, 0.3});
        mesh.setGridGlobalOffset(std::vector<double>{0.0, 1.0, -2.0});
        mesh.setAttribute("gridUnitSI", 1.0);
        mesh.setAttribute("sdf_axis_x", std::vector<double>{0, 0.1, 0.2});
        mesh.setUnitDimension({{UnitDimension::L, 1}, {UnitDimension::M, 1},
            {UnitDimension::T, -3}, {UnitDimension::I, -1}});

        int64_t const nx = 8, ny = 6, nz = 5;
        int64_t const base = nx / size, rem = nx % size;
        int64_t const cnt = rank < rem ? base + 1 : base;
        int64_t const off0 = rank < rem ? rank * (base + 1)
                                        : rem * (base + 1)
                                              + (rank - rem) * base;

        for (auto comp : {"x", "y", "z"})
        {
            RecordComponent & rc = mesh[comp];
            rc.resetDataset(
                Dataset(Datatype::DOUBLE, Extent{nx, ny, nz}));
            auto buf = std::unique_ptr<double[]>(new double[cnt * ny * nz]);
            for (int64_t i = 0; i < cnt * ny * nz; ++i)
                buf[i] = double(i + step);
            if (cnt > 0)
                rc.storeChunk(std::move(buf), Offset{uint64_t(off0), 0, 0},
                    Extent{uint64_t(cnt), uint64_t(ny), uint64_t(nz)});
            rc.setAttribute("sdf_id", std::string(comp));
        }
        series.flush();

        // particles: 103 split across ranks
        if (!noParticles)
        {
        int64_t const np = 103;
        int64_t const pbase = np / size, prem = np % size;
        int64_t const pcnt = rank < prem ? pbase + 1 : pbase;
        int64_t const poff = rank < prem ? rank * (pbase + 1)
                                         : prem * (pbase + 1)
                                               + (rank - prem) * pbase;

        ParticleSpecies e = it.particles["el"];
        for (auto comp : {"x", "y", "z"})
        {
            RecordComponent & rc = e["position"][comp];
            rc.resetDataset(Dataset(Datatype::DOUBLE, Extent{uint64_t(np)}));
            auto buf = std::unique_ptr<double[]>(new double[pcnt]);
            for (int64_t i = 0; i < pcnt; ++i)
                buf[i] = 0.5 * double(poff + i);
            if (pcnt > 0)
                rc.storeChunk(std::move(buf), Offset{uint64_t(poff)},
                    Extent{uint64_t(pcnt)});
        }
        RecordComponent & w = e["weighting"][RecordComponent::SCALAR];
        w.resetDataset(Dataset(Datatype::DOUBLE, Extent{uint64_t(np)}));
        {
            auto buf = std::unique_ptr<double[]>(new double[pcnt]);
            for (int64_t i = 0; i < pcnt; ++i)
                buf[i] = 4.0;
            if (pcnt > 0)
                w.storeChunk(std::move(buf), Offset{uint64_t(poff)},
                    Extent{uint64_t(pcnt)});
        }
        e["position"].setUnitDimension({{UnitDimension::L, 1}});
        series.flush();
        }
        else
        {
            // replicate converter mesh extras (int/string attrs, position)
            mesh.setAttribute("sdf_mesh_id", std::string("general_grid"));
            mesh.setAttribute("position", std::vector<double>{0.0, 0.0, 0.5});
            mesh["x"].setAttribute("sdf_id", std::string("ex"));
            mesh["x"].setAttribute("sdf_stagger", 1);
            mesh["x"].setAttribute("sdf_units", std::string("V/m"));
            it.setAttribute("sdf_nsteps", int64_t(10));
            it.setAttribute("sdf_cfl", 0.95);
        }

        it.close();
    }
    series.flush();

    if (rank == 0)
        std::printf("STRESS OK\n");
    MPI_Finalize();
    return 0;
}
