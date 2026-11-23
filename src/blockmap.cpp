#include "blockmap.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <unordered_map>

namespace sdf2opmd
{
namespace
{
std::string
trimmed(std::string s)
{
    auto notspace = [](unsigned char c) { return c != ' ' && c != '\0'; };
    s.erase(s.begin(),
        std::find_if(s.begin(), s.end(), notspace)); // NOLINT
    s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    return s;
}

std::string
str(std::string const & s)
{
    return trimmed(s);
}

MetaValue
constantMeta(sdf_block_t * b)
{
    MetaValue mv;
    mv.name = str(b->id ? b->id : "");
    switch (b->datatype)
    {
    case SDF_DATATYPE_REAL8:
        mv.value = *reinterpret_cast<double const *>(b->const_value);
        break;
    case SDF_DATATYPE_REAL4:
        mv.value = double(*reinterpret_cast<float const *>(b->const_value));
        break;
    case SDF_DATATYPE_INTEGER4:
        mv.value = *reinterpret_cast<int32_t const *>(b->const_value);
        break;
    case SDF_DATATYPE_INTEGER8:
        mv.value = *reinterpret_cast<int64_t const *>(b->const_value);
        break;
    case SDF_DATATYPE_LOGICAL:
        mv.value = bool(b->const_value[0] != 0);
        break;
    case SDF_DATATYPE_CHARACTER: {
        std::string s(b->const_value,
            strnlen(b->const_value, sizeof(b->const_value)));
        mv.value = str(s);
        break;
    }
    default:
        mv.value = std::string("<unsupported datatype>");
        break;
    }
    return mv;
}

void
nameValueMeta(sdf_block_t * b, std::vector<MetaValue> & out)
{
    int const n = b->nmaterial_names;
    if (n <= 0 || !b->material_names || !b->data)
        return;

    auto push = [&](int i, MetaValue mv) {
        mv.name = str(b->material_names[i]);
        out.push_back(std::move(mv));
    };

    for (int i = 0; i < n; ++i)
    {
        MetaValue mv;
        switch (b->datatype)
        {
        case SDF_DATATYPE_REAL8:
            mv.value = reinterpret_cast<double *>(b->data)[i];
            break;
        case SDF_DATATYPE_REAL4:
            mv.value = double(reinterpret_cast<float *>(b->data)[i]);
            break;
        case SDF_DATATYPE_INTEGER4:
            mv.value = reinterpret_cast<int32_t *>(b->data)[i];
            break;
        case SDF_DATATYPE_INTEGER8:
            mv.value = reinterpret_cast<int64_t *>(b->data)[i];
            break;
        case SDF_DATATYPE_LOGICAL:
            mv.value = bool(reinterpret_cast<char *>(b->data)[i] != 0);
            break;
        case SDF_DATATYPE_CHARACTER: {
            auto ** strs = reinterpret_cast<char **>(b->data);
            mv.value = str(strs[i] ? strs[i] : "");
            break;
        }
        default:
            continue;
        }
        push(i, std::move(mv));
    }
}

bool
isSupportedFieldDtype(int32_t t)
{
    return t == SDF_DATATYPE_REAL4 || t == SDF_DATATYPE_REAL8
        || t == SDF_DATATYPE_INTEGER4 || t == SDF_DATATYPE_INTEGER8;
}

std::string
typeToString(int32_t t)
{
    if (t >= 0 && t <= SDF_BLOCKTYPE_NAMEVALUE)
        return sdf_blocktype_c[t];
    return "unknown";
}

} // namespace

BlockMap
classify(std::vector<sdf_block_t *> const & blocks)
{
    BlockMap map;

    std::unordered_map<std::string, sdf_block_t *> meshById;

    // First pass: grids, species meshes, fields, metadata
    for (auto * b : blocks)
    {
        switch (b->blocktype)
        {
        case SDF_BLOCKTYPE_PLAIN_MESH:
            map.meshes.push_back(b);
            meshById[str(b->id)] = b;
            break;

        case SDF_BLOCKTYPE_POINT_MESH: {
            std::string sp =
                b->material_id ? str(b->material_id) : std::string();
            auto & sd = map.species[sp];
            sd.species = sp;
            if (!sd.mesh)
                sd.mesh = b;
            break;
        }

        case SDF_BLOCKTYPE_PLAIN_VARIABLE:
            if (isSupportedFieldDtype(b->datatype))
                map.fields.push_back(b);
            else
                map.skipped.push_back("field '" + str(b->id)
                    + "': unsupported datatype "
                    + sdf_datatype_c[b->datatype]);
            break;

        case SDF_BLOCKTYPE_CONSTANT: {
            auto mv = constantMeta(b);
            std::string lname = mv.name;
            std::transform(lname.begin(), lname.end(), lname.begin(),
                [](unsigned char c) { return std::tolower(c); });
            if (lname == "dt")
            {
                if (auto const * d = std::get_if<double>(&mv.value))
                {
                    map.dt = *d;
                    map.hasDt = true;
                }
            }
            map.constants.push_back(std::move(mv));
            break;
        }

        case SDF_BLOCKTYPE_NAMEVALUE:
            nameValueMeta(b, map.constants);
            break;

        case SDF_BLOCKTYPE_SCRUBBED:
        case SDF_BLOCKTYPE_NULL:
            break;

        default:
            break; // second pass / point variables below
        }
    }

    // Second pass: point variables (need species meshes known already)
    for (auto * b : blocks)
    {
        if (b->blocktype != SDF_BLOCKTYPE_POINT_VARIABLE)
            continue;

        std::string sp = b->material_id ? str(b->material_id) : std::string();
        if (sp.empty() && b->mesh_id)
        {
            // Pre-1.3 files: take species from the referenced point mesh
            auto * mesh = meshById[str(b->mesh_id)];
            if (mesh && mesh->material_id)
                sp = str(mesh->material_id);
        }
        if (sp.empty())
            sp = "unknown";

        auto & sd = map.species[sp];
        sd.species = sp;
        if (!sd.mesh && b->mesh_id)
            sd.mesh = meshById[str(b->mesh_id)];
        sd.vars.push_back(b);
    }

    // Notes for everything we deliberately do not convert (v1 scope)
    static int const skippedTypes[] = {
        SDF_BLOCKTYPE_STITCHED,
        SDF_BLOCKTYPE_STITCHED_TENSOR,
        SDF_BLOCKTYPE_STITCHED_MATERIAL,
        SDF_BLOCKTYPE_STITCHED_MATVAR,
        SDF_BLOCKTYPE_STITCHED_SPECIES,
        SDF_BLOCKTYPE_STITCHED_OBSTACLE_GROUP,
        SDF_BLOCKTYPE_CONTIGUOUS,
        SDF_BLOCKTYPE_CONTIGUOUS_TENSOR,
        SDF_BLOCKTYPE_CONTIGUOUS_MATERIAL,
        SDF_BLOCKTYPE_CONTIGUOUS_MATVAR,
        SDF_BLOCKTYPE_CONTIGUOUS_SPECIES,
        SDF_BLOCKTYPE_SPECIES,
        SDF_BLOCKTYPE_LAGRANGIAN_MESH,
        SDF_BLOCKTYPE_UNSTRUCTURED_MESH,
        SDF_BLOCKTYPE_STATION,
        SDF_BLOCKTYPE_SOURCE,
        SDF_BLOCKTYPE_DATABLOCK,
        SDF_BLOCKTYPE_RUN_INFO,
        SDF_BLOCKTYPE_CPU_SPLIT,
        SDF_BLOCKTYPE_ARRAY,
    };
    for (auto * b : blocks)
    {
        for (int t : skippedTypes)
        {
            if (b->blocktype == t)
            {
                map.skipped.push_back("skipped block '" + str(b->id) + "' ("
                    + typeToString(t) + ", not converted in v1)");
                break;
            }
        }
    }

    return map;
}

} // namespace sdf2opmd
