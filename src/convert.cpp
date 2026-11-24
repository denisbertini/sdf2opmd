#include "convert.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <sys/resource.h>
#include <unordered_map>

namespace fs = std::filesystem;
using namespace openPMD;

namespace sdf2opmd
{
namespace
{
/* Trim trailing/leading space padding used by SDF fixed-width strings. */
std::string
trim(std::string s)
{
    auto notspace = [](unsigned char c) { return c != ' ' && c != '\0'; };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    return s;
}

std::string
trim(sdf_block_t const * b, char const * s)
{
    (void)b;
    return trim(std::string(s ? s : ""));
}

void
setMetaAttr(Attributable & a, std::string const & name, MetaValue const & mv)
{
    std::visit(
        [&](auto const & v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, int32_t>)
                a.setAttribute(name, int64_t(v));
            else
                a.setAttribute(name, v);
        },
        mv.value);
}

/* Uniform-axis geometry of one SDF plain mesh. */
struct Axis
{
    std::vector<double> nodes;
    double spacing = 1.0;
    double offset = 0.0;
    bool uniform = true;
    std::string label;
};

struct MeshGeom
{
    std::vector<Axis> axes; // size = ndims
    int ndims = 3;
    bool valid = false;
};

MeshGeom
readMeshGeom(SdfFile & file, sdf_block_t * mb)
{
    MeshGeom geom;
    if (!mb)
    {
        geom.ndims = 3;
        geom.axes.resize(3);
        for (int d = 0; d < 3; ++d)
            geom.axes[d].label = std::string(1, char('x' + d));
        return geom; // unit spacing defaults
    }

    geom.ndims = mb->ndims;
    geom.axes.resize(geom.ndims);
    geom.valid = true;

    auto nodes = [&] {
        if (mb->datatype == SDF_DATATYPE_REAL4)
        {
            auto f4 = readWhole<float>(file, *mb);
            return std::vector<double>(f4.begin(), f4.end());
        }
        return readWhole<double>(file, *mb);
    }();

    size_t pos = 0;
    for (int d = 0; d < geom.ndims; ++d)
    {
        auto const len = size_t(mb->dims[d]);
        Axis & ax = geom.axes[d];
        if (pos + len > nodes.size())
            throw std::runtime_error("mesh block data shorter than dims");
        ax.nodes.assign(nodes.begin() + pos, nodes.begin() + pos + len);
        pos += len;

        if (mb->dim_labels && d < mb->ndim_labels && mb->dim_labels[d])
            ax.label = trim(mb->dim_labels[d]);
        if (ax.label.empty())
            ax.label = std::string(1, char('x' + d));

        if (len > 1)
        {
            ax.offset = ax.nodes.front();
            ax.spacing = (ax.nodes.back() - ax.nodes.front())
                / double(len - 1);
            if (ax.spacing == 0.0)
                throw std::runtime_error("degenerate grid axis");
            double const tol = 1e-6 * std::abs(ax.spacing);
            for (size_t i = 1; i < ax.nodes.size(); ++i)
            {
                if (std::abs((ax.nodes[i] - ax.nodes[i - 1]) - ax.spacing)
                    > tol)
                {
                    ax.uniform = false;
                    break;
                }
            }
        }
    }
    return geom;
}

// process RSS high-water mark in MiB (Linux ru_maxrss is KiB)
double
peakRssMiB()
{
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0)
        return 0.0;
    return double(ru.ru_maxrss) / 1024.0;
}

void
applyGeom(Mesh & mesh, MeshGeom const & geom, std::string const & sdfMeshId)
{
    int const nd = geom.ndims;
    std::vector<double> spacing(nd), offset(nd);
    std::vector<std::string> labels(nd);
    bool uniform = true;
    for (int d = 0; d < nd; ++d)
    {
        spacing[d] = geom.axes[d].spacing;
        offset[d] = geom.axes[d].offset;
        labels[d] = geom.axes[d].label;
        uniform = uniform && geom.axes[d].uniform;
    }
    mesh.setGeometry(Mesh::Geometry::cartesian);
    mesh.setAxisLabels(labels);
    mesh.setGridSpacing(spacing);
    mesh.setGridGlobalOffset(offset);
    mesh.setAttribute("gridUnitSI", 1.0);
    if (!uniform)
        mesh.setAttribute("sdf_nonuniform_grid", true);
    if (!sdfMeshId.empty())
        mesh.setAttribute("sdf_mesh_id", sdfMeshId);

    // Full axes for non-uniform or audit-hungry consumers (small cost).
    for (int d = 0; d < nd; ++d)
    {
        auto const & ax = geom.axes[d];
        if (!ax.nodes.empty() && ax.nodes.size() <= (1u << 22))
            mesh.setAttribute("sdf_axis_" + ax.label, ax.nodes);
    }
}

} // namespace

void
Timings::print(MPI_Comm comm) const
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // gather honest per-rank memory high-water marks (RSS includes the
    // allocator's arena growth: exactly what SLURM mem-per-cpu must cover)
    double peakHere = peakRssMiB;
    if (size > 1)
        MPI_Allreduce(MPI_IN_PLACE, &peakHere, 1, MPI_DOUBLE, MPI_MAX, comm);
    double peakSum = peakRssMiB;
    if (size > 1)
        MPI_Allreduce(MPI_IN_PLACE, &peakSum, 1, MPI_DOUBLE, MPI_SUM, comm);

    if (rank != 0 || files == 0)
        return;
    auto per = [this](double v) { return v / double(files); };
    std::printf(
        "\nsdf2opmd timing over %d file(s): total %.3f s "
        "[metadata %.3f s | data reads %.3f s | engine flush %.3f s] "
        "(%.3f s/file)\n",
        files, total, per(meta), per(read), per(flush),
        per(total));
    std::printf(
        "sdf2opmd memory: peak RSS/rank %.0f MiB (max over %d rank(s), "
        "mean %.0f MiB)\n",
        peakHere, size, peakSum / double(size));
}

Series
makeSeries(Cli const & cli, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    fs::path out = cli.outdir.empty() ? fs::path(".") : fs::path(cli.outdir);
    if (rank == 0)
        fs::create_directories(out);
    MPI_Barrier(comm);

    std::string path = (out / cli.name).string();
    std::string const config =
        cli.engine.empty() ? std::string("{}") : cli.engine;

    Series series(path, Access::CREATE, comm, config);
    series.setIterationEncoding(IterationEncoding::fileBased);
    series.setAuthor(cli.author);
    if (!cli.machine.empty())
        series.setMachine(cli.machine);
    series.setSoftware("sdf2opmd");
    series.setSoftwareVersion(std::string(SDF2OPMD_VERSION) + " + SDF-C "
        + sdf_get_library_commit_id());
    return series;
}

void
printInfo(SdfFile & file, std::string const & path, BlockMap const & map)
{
    int rank = 0;
    MPI_Comm_rank(file.comm(), &rank);
    if (rank != 0)
        return;

    std::printf("\n%s\n  step %d  time %.9g  code '%s'  blocks %d\n",
        path.c_str(), file.step(), file.time(), file.codeName().c_str(),
        int(file.blocks().size()));

    for (auto * b : file.blocks())
    {
        if (b->blocktype == SDF_BLOCKTYPE_SCRUBBED)
            continue;
        std::printf("  %-8.8s %-14.14s", b->id ? b->id : "?",
            (b->blocktype >= 0
                        && b->blocktype <= SDF_BLOCKTYPE_NAMEVALUE)
                ? sdf_blocktype_c[b->blocktype]
                : "?");
        std::printf(" %-8.8s nd=%d dims=",
            (b->datatype >= 0 && b->datatype <= SDF_DATATYPE_OTHER)
                ? sdf_datatype_c[b->datatype]
                : "?",
            b->ndims);
        for (int d = 0; d < b->ndims; ++d)
            std::printf("%" PRId64 "%s", b->dims[d],
                d + 1 < b->ndims ? "x" : "");
        if (b->mesh_id && *b->mesh_id)
            std::printf("  mesh=%s", b->mesh_id);
        if (b->material_id && *b->material_id)
            std::printf("  species=%s", b->material_id);
        std::printf("  bytes=%" PRId64 "\n", b->data_length);
    }

    for (auto const & note : map.skipped)
        std::printf("  note: %s\n", note.c_str());
    std::printf("\n");
}

void
convertFile(Series & series, SdfFile & file, std::string const & path,
    Cli const & cli, Timings & timings)
{
    auto const t0 = now();
    CommSize const cs(file.comm());
    int const rank = cs.rank;
    auto const step = uint64_t(file.step());

    auto const map = classify(file.blocks());
    if (!cli.quiet && rank == 0)
    {
        // per-rank payload estimate: fields are batched and particles held
        // until their flush, so one rank roughly balances total_data/nranks
        int64_t dataBytes = 0;
        for (auto * b : file.blocks())
            if (b->blocktype != SDF_BLOCKTYPE_SCRUBBED)
                dataBytes += b->data_length;
        std::printf(
            "converting %s -> iteration %" PRIu64
            " (%zu field block(s)",
            path.c_str(), step, map.fields.size());
        if (!map.species.empty())
            std::printf(", %zu species", map.species.size());
        std::printf("): %.1f MiB data, ~%.1f MiB/rank at %d ranks\n",
            double(dataBytes) / (1024.0 * 1024.0),
            double(dataBytes) / (1024.0 * 1024.0 * double(cs.size)),
            cs.size);
        for (auto const & note : map.skipped)
            std::printf("  note: %s\n", note.c_str());
    }

    Iteration it = series.iterations[step];
    it.setTime(file.time());
    it.setTimeUnitSI(1.0); // EPOCH writes SI time
    if (map.hasDt)
        it.setDt(map.dt);
    it.setAttribute("sdf_source", path);

    for (auto const & mv : map.constants)
    {
        std::string lower = mv.name;
        std::transform(lower.begin(), lower.end(), lower.begin(),
            [](unsigned char c) { return std::tolower(c); });
        if (lower == "dt")
            continue; // consumed by Iteration::setDt
        setMetaAttr(it, "sdf_" + sanitize(mv.name), mv);
    }

    std::unordered_map<std::string, sdf_block_t *> meshById;
    for (auto * m : map.meshes)
        meshById[trim(m->id)] = m;

    /* ------------------------------------------------------------------ */
    /* Fields: SDF plain variables -> openPMD records                      */
    /*                                                                     */
    /* openPMD records share extent, grid geometry and position across all */
    /* their components. Only SDF variables with identical dims *and*      */
    /* stagger may therefore form one vector record (e.g. a fully          */
    /* consistent E/x,y,z). EPOCH's Yee components differ in both, so they */
    /* come out as scalar records named by the SDF id, each carrying its   */
    /* exact grid `position` derived from the stagger.                     */
    /* ------------------------------------------------------------------ */
    std::vector<sdf_block_t *> selected;
    for (auto * b : map.fields)
    {
        std::string const id = trim(b->id);
        if (!cli.fields.empty()
            && std::find(cli.fields.begin(), cli.fields.end(), id)
                == cli.fields.end())
        {
            continue;
        }
        selected.push_back(b);
    }

    auto groupKey = [](sdf_block_t const * b, FieldTarget const & tgt) {
        std::string k = tgt.mesh;
        k += "|nd=" + std::to_string(b->ndims);
        for (int d = 0; d < b->ndims; ++d)
            k += "|d" + std::to_string(d) + "=" + std::to_string(b->dims[d]);
        k += "|st=" + std::to_string(b->stagger);
        k += "|m=" + trim(b->mesh_id);
        return k;
    };

    std::map<std::string, std::vector<sdf_block_t *>> groups;
    std::vector<std::string> groupOrder;
    for (auto * b : selected)
    {
        std::string k = groupKey(b, fieldTarget(trim(b->id)));
        if (!groups.count(k))
            groupOrder.push_back(k);
        groups[k].push_back(b);
    }

    struct PlannedField
    {
        std::string record;
        std::string component; // "" == SCALAR
        sdf_block_t * block;
    };
    std::vector<PlannedField> plan;
    std::set<std::string> usedRecords;
    for (auto const & key : groupOrder)
    {
        auto const & blks = groups[key];
        FieldTarget const first = fieldTarget(trim(blks.front()->id));
        if (!first.component.empty() && blks.size() > 1)
        {
            std::string rec = first.mesh;
            if (!usedRecords.insert(rec).second)
                rec = first.mesh + "_" + sanitize(trim(blks.front()->mesh_id));
            for (auto * b : blks)
                plan.push_back({rec, fieldTarget(trim(b->id)).component, b});
        }
        else
        {
            for (auto * b : blks)
            {
                std::string rec = sanitize(trim(b->id));
                usedRecords.insert(rec);
                plan.push_back({rec, "", b});
            }
        }
    }

    std::map<std::string, MeshGeom> geomCache;
    auto geomFor = [&](std::string const & meshId) -> MeshGeom const & {
        auto git = geomCache.find(meshId);
        if (git != geomCache.end())
            return git->second;
        auto mit = meshById.find(meshId);
        MeshGeom g = readMeshGeom(
            file, mit == meshById.end() ? nullptr : mit->second);
        return geomCache.emplace(meshId, std::move(g)).first->second;
    };

    auto const tf = now();
    std::set<std::string> recInit;
    for (auto const & pf : plan)
    {
        sdf_block_t * b = pf.block;
        std::string const id = trim(b->id);
        std::string const sdfMeshId = trim(b->mesh_id);

        Mesh mesh = it.meshes[pf.record];
        if (!recInit.count(pf.record))
        {
            applyGeom(mesh, geomFor(sdfMeshId), sdfMeshId);
            std::vector<double> pos;
            if (staggerToPosition(b->stagger, b->ndims, pos))
                mesh.setAttribute("position", pos);
            std::map<UnitDimension, double> ud;
            if (fieldUnitDimension(fieldTarget(id).mesh, ud)
                || fieldUnitDimension(id, ud))
            {
                mesh.setUnitDimension(ud);
            }
            recInit.insert(pf.record);
        }

        RecordComponent & rc = mesh[pf.component.empty()
                ? MeshRecordComponent::SCALAR
                : pf.component];

        int64_t counts[SDF_MAXDIMS];
        int64_t starts[SDF_MAXDIMS];
        int const nd = b->ndims;
        Slab const slab = evenSplit(b->dims[0], cs.size, rank);
        starts[0] = slab.start;
        counts[0] = slab.count;
        for (int d = 1; d < nd; ++d)
        {
            starts[d] = 0;
            counts[d] = b->dims[d];
        }

        withDtype(b->datatype, [&]<typename T>() {
            Extent extent;
            for (int d = 0; d < nd; ++d)
                extent.push_back(uint64_t(b->dims[d]));
            rc.resetDataset(Dataset(determineDatatype<T>(), extent));

            // All ranks read (collective), even the empty ones.
            auto const tr = now();
            auto buf = readFieldSlab<T>(file, *b, starts, counts);
            timings.read += now() - tr;

            if (slab.count > 0)
            {
                Offset off;
                Extent ext;
                for (int d = 0; d < nd; ++d)
                {
                    off.push_back(uint64_t(starts[d]));
                    ext.push_back(uint64_t(counts[d]));
                }
                rc.storeChunk(std::move(buf), std::move(off), std::move(ext));
            }
        });

        rc.setAttribute("sdf_id", id);
        rc.setAttribute("sdf_stagger", int(b->stagger));
        std::string const units = trim(b->units);
        if (!units.empty())
            rc.setAttribute("sdf_units", units);
        if (b->mult != 0.0 && b->mult != 1.0)
            rc.setAttribute("sdf_multiplier", b->mult);
    }
    timings.meta += tf - t0;

    {
        auto const t = now();
        series.flush();
        timings.flush += now() - t;
    }

    /* ------------------------------------------------------------------ */
    /* Particles: SDF point data -> openPMD particle records               */
    /* ------------------------------------------------------------------ */
    for (auto const & [name, sd] : map.species)
    {
        std::string const species = trim(name.c_str());
        if (!cli.species.empty()
            && std::find(cli.species.begin(), cli.species.end(), species)
                == cli.species.end())
        {
            continue;
        }
        if (sd.vars.empty())
            continue;

        int64_t npart = 0;
        if (sd.mesh)
            npart = sd.mesh->nelements;
        for (auto * v : sd.vars)
            npart = std::max(npart, v->nelements);

        ParticleSpecies e = it.particles[sanitize(species)];
        e.setAttribute("sdf_species", species);

        Slab const slab = evenSplit(npart, cs.size, rank);
        std::set<std::string> unitDone;

        for (auto * v : sd.vars)
        {
            if (v->datatype != SDF_DATATYPE_REAL8
                && v->datatype != SDF_DATATYPE_REAL4
                && v->datatype != SDF_DATATYPE_INTEGER4
                && v->datatype != SDF_DATATYPE_INTEGER8)
            {
                if (rank == 0)
                    std::printf("  note: skipping particle variable '%s' "
                                "(unsupported datatype)\n",
                        trim(v->id).c_str());
                continue;
            }

            PartTarget const pt = particleTarget(trim(v->id), species);
            RecordComponent & rc =
                pt.component.empty() ? e[pt.record][RecordComponent::SCALAR]
                                     : e[pt.record][pt.component];

            if (!unitDone.count(pt.record))
            {
                if (std::map<UnitDimension, double> ud;
                    particleUnitDimension(pt.record, ud))
                {
                    e[pt.record].setUnitDimension(ud);
                }
                e[pt.record].setUnitSI(1.0); // EPOCH particle data is SI
                unitDone.insert(pt.record);
            }

            withDtype(v->datatype, [&]<typename T>() {
                rc.resetDataset(
                    Dataset(determineDatatype<T>(), Extent{uint64_t(npart)}));
                auto const tr = now();
                auto buf =
                    readPointSlab<T>(file, *v, slab.start, slab.count);
                timings.read += now() - tr;
                if (slab.count > 0)
                {
                    rc.storeChunk(std::move(buf), Offset{uint64_t(slab.start)},
                        Extent{uint64_t(slab.count)});
                }
            });
            rc.setAttribute("sdf_id", trim(v->id));
        }

        auto const t = now();
        series.flush();
        timings.flush += now() - t;
    }

    it.close();
    timings.total += now() - t0;
    ++timings.files;
    {
        double const rss = peakRssMiB();
        if (rss > timings.peakRssMiB)
            timings.peakRssMiB = rss;
    }
}

} // namespace sdf2opmd
