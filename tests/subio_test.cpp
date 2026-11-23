//
// Standalone repro probe for the field-read MPI-IO pattern used by
// sdf2opmd's readFieldSlab: Fortran-ordered subarray view + read_all.
//
// build: mpicxx -O2 -o subio_test subio_test.cpp -lm
//
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

int
main(int argc, char * * argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    char const * file = "/tmp/subio_test.dat";
    int64_t const nx = 8, ny = 6, nz = 5;
    int64_t const total = nx * ny * nz;
    int64_t const disp = 4856;

    MPI_File fh;
    MPI_File_open(comm, (char *)file,
        MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);

    // rank 0 fills the file with recognizable data
    if (rank == 0)
    {
        std::vector<double> data(total);
        for (int64_t k = 0; k < nz; ++k)
            for (int64_t j = 0; j < ny; ++j)
                for (int64_t i = 0; i < nx; ++i)
                    data[i + nx * (j + ny * k)]
                        = 1000000 * k + 1000 * j + i;
        MPI_Status st;
        std::vector<char> pad(disp, 0);
        MPI_File_write_at(fh, 0, pad.data(), (int)pad.size(), MPI_BYTE, &st);
        MPI_File_write_at(
            fh, disp, data.data(), (int)(total * 8), MPI_BYTE, &st);
    }
    MPI_Barrier(comm);
    

    // even split of x over ranks (same as evenSplit())
    int64_t const base = nx / size, rem = nx % size;
    int64_t const cnt = rank < rem ? base + 1 : base;
    int64_t const start = rank < rem
        ? rank * (base + 1)
        : rem * (base + 1) + (rank - rem) * base;

    int g[3] = {(int)nx, (int)ny, (int)nz};
    int l[3] = {(int)cnt, (int)ny, (int)nz};
    int s[3] = {(int)start, 0, 0};

    enum Mode { MODE_F = 0, MODE_C = 1, MODE_HV = 2 };
    static int mode = [] {
        const char * m = getenv("SUBIO_MODE");
        return m ? atoi(m) : MODE_F;
    }();

    for (int iter = 0; iter < 50; ++iter)
    {
        MPI_Datatype type;
        if (mode == MODE_F)
        {
            MPI_Type_create_subarray(3, g, l, s, MPI_ORDER_FORTRAN,
                MPI_DOUBLE, &type);
        }
        else if (mode == MODE_C)
        {
            int gc[3] = {(int)nz, (int)ny, (int)nx};
            int lc[3] = {(int)nz, (int)ny, (int)cnt};
            int sc[3] = {0, 0, (int)start};
            MPI_Type_create_subarray(3, gc, lc, sc, MPI_ORDER_C, MPI_DOUBLE,
                &type);
        }
        else
        {
            MPI_Datatype col, plane;
            MPI_Type_vector((int)cnt, 1, nx, MPI_DOUBLE, &col);
            MPI_Type_vector((int)ny, 1, nx * ny, col, &plane);
            MPI_Type_vector((int)nz, 1, nx * ny * nz, plane, &type);
            MPI_Type_free(&col);
            MPI_Type_free(&plane);
        }
        MPI_Type_commit(&type);

        std::vector<double> buf(std::max<int64_t>(cnt * ny * nz, 1), -1.0);

        MPI_Offset view
            = (mode == MODE_HV) ? (MPI_Offset)(disp + start * 8)
                                : (MPI_Offset)disp;
        int e1 = MPI_File_set_view(
            fh, view, MPI_DOUBLE, type, "native", MPI_INFO_NULL);
        int e2 = getenv("SUBIO_FIX")
            ? MPI_File_read_all(fh, buf.data(), (int)buf.size(), MPI_DOUBLE,
                  MPI_STATUS_IGNORE)
            : MPI_File_read_all(fh, buf.data(), 1, type, MPI_STATUS_IGNORE);
        int e3 = MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native",
            MPI_INFO_NULL);
        MPI_Type_free(&type);

        // MODE_F/C map: buf[i + cnt*(j + ny*k)]; MODE_HV map is k-major
        double err = 0;
        for (int64_t i = 0; i < cnt; ++i)
            for (int64_t j = 0; j < ny; ++j)
                for (int64_t k = 0; k < nz; ++k)
                {
                    double want = 1000000.0 * k + 1000.0 * j
                        + double(start + i);
                    double got = mode == MODE_HV
                        ? buf[(k * ny + j) * cnt + i]
                        : buf[i + cnt * (j + ny * k)];
                    if (got != want)
                        err += 1;
                }
        if (iter == 0 || err != 0)
            std::printf("mode%d r%d iter%d errs=%d mpi=%d,%d,%d got[%.0f %.0f "
                        "%.0f %.0f]\n",
                mode, rank, iter, (int)err, e1, e2, e3, buf[0], buf[1],
                buf[cnt], buf[std::min<int64_t>(cnt * ny,
                                 (int64_t)buf.size() - 1)]);
    }

    
    MPI_File_close(&fh);
    if (rank == 0)
        std::printf("SUBIO DONE\n");
    MPI_Finalize();
    return 0;
}
