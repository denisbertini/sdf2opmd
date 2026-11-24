//
// sdf2opmd - convert EPOCH/SDF snapshots to openPMD (ADIOS2 BP5).
//
// Fast-path design:
//   * SDF_C parses each file's metadata exactly once (header + summary ->
//     hash-indexed blocklist), collectively.
//   * Field data is read through Fortran-ordered MPI-IO subarray views
//     (subarray filetype + contiguous buffer read, the monotonic/fast ROMIO
//     path) and turned into C-order chunks by a cache-blocked per-rank
//     transpose.
//   * Buffers are handed to openPMD with ownership (storeChunk of unique_ptr)
//     and flushed in batches: one collective engine flush per field batch and
//     per species, one per file minimum.
//
#include <exception>
#include <filesystem>
#include <iostream>

#include <mpi.h>

#include "cli.hpp"
#include "convert.hpp"
#include "mpi.hpp"
#include "sdf_file.hpp"

namespace
{
int
fail(MPI_Comm comm, std::string const & msg)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
        std::cerr << "sdf2opmd: " << msg << std::endl;
    int flag = 1;
    MPI_Finalized(&flag);
    if (!flag)
        MPI_Abort(comm, 1);
    return 1;
}
} // namespace

int
main(int argc, char * * argv)
{
    using namespace sdf2opmd;

    Cli cli;
    try
    {
        cli = Cli::parse(argc, argv);
    }
    catch (std::exception const & e)
    {
        std::cerr << "sdf2opmd: " << e.what() << std::endl;
        return 2;
    }

    MpiScope mpi(argc, &argv);
    CommSize const cs(MPI_COMM_WORLD);

    try
    {
        auto const files = cli.resolveFiles();

        // rank 0 checks readability, broadcasts the verdict
        int ok = 1;
        if (cs.rank == 0)
        {
            for (auto const & f : files)
            {
                if (!std::filesystem::exists(f))
                {
                    std::cerr << "sdf2opmd: missing file: " << f << std::endl;
                    ok = 0;
                }
            }
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok)
            throw std::runtime_error("input files missing");

        if (cli.info)
        {
            for (auto const & f : files)
            {
                SdfFile file = SdfFile::open(MPI_COMM_WORLD, f);
                auto map = classify(file.blocks());
                printInfo(file, f, map);
            }
            return 0;
        }

        auto series = makeSeries(cli, MPI_COMM_WORLD);

        Timings timings;
        for (auto const & path : files)
        {
            SdfFile file = SdfFile::open(MPI_COMM_WORLD, path);
            if (series.iterations.count(uint64_t(file.step())) != 0)
            {
                if (cs.rank == 0)
                    std::printf(
                        "sdf2opmd: note: skipping %s, iteration %d already "
                        "converted\n",
                        path.c_str(), file.step());
                continue; // SdfFile closes collectively here
            }

            convertFile(series, file, path, cli, timings);
            // file closes here (collective inside SDF_C)
        }

        timings.print(MPI_COMM_WORLD);
        if (cs.rank == 0 && !cli.quiet)
            std::printf("done.\n");
    }
    catch (std::exception const & e)
    {
        return fail(MPI_COMM_WORLD, e.what());
    }

    return 0;
}
