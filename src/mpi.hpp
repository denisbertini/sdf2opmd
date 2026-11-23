#pragma once
//
// Tiny MPI conveniences: init/finalize guard and error checking.
//
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sdf2opmd
{
inline void
mpiCheck(int err, char const * what)
{
    if (err != MPI_SUCCESS)
    {
        int len = 0;
        char buf[MPI_MAX_ERROR_STRING] = {0};
        MPI_Error_string(err, buf, &len);
        throw std::runtime_error(std::string(what) + ": " + buf);
    }
}

class MpiScope
{
public:
    MpiScope(int & argc, char ** * argv)
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            int provided = MPI_THREAD_SINGLE;
            mpiCheck(MPI_Init_thread(&argc, argv, MPI_THREAD_SERIALIZED,
                        &provided),
                "MPI_Init_thread");
            owned_ = true;
        }
    }

    MpiScope(MpiScope const &) = delete;
    MpiScope & operator=(MpiScope const &) = delete;

    ~MpiScope()
    {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (owned_ && !finalized)
            MPI_Finalize();
    }

private:
    bool owned_ = false;
};

struct CommSize
{
    int rank = 0;
    int size = 1;
    explicit CommSize(MPI_Comm comm = MPI_COMM_WORLD)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }
};

} // namespace sdf2opmd
