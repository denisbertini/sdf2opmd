#pragma once
//
// One pass over an SDF blocklist, classifying blocks into the categories the
// converter understands, and extracting scalar metadata (constants,
// name/value pairs) into plain values. Unknown block types are reported and
// skipped, never fatal.
//
#include <sdf.h>

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace sdf2opmd
{
struct MetaValue
{
    std::string name;
    std::variant<bool, int32_t, int64_t, double, std::string> value;
};

struct SpeciesData
{
    std::string species;            // SDF species id
    sdf_block_t * mesh = nullptr;   // associated point mesh (may be null)
    std::vector<sdf_block_t *> vars; // point variables, file order
};

struct BlockMap
{
    std::vector<sdf_block_t *> meshes;   // SDF plain meshes (grids)
    std::vector<sdf_block_t *> fields;   // SDF plain variables
    std::map<std::string, SpeciesData> species;
    std::vector<MetaValue> constants;    // constants + namevalue scalars
    std::vector<std::string> skipped;    // human-readable notes

    // The SDF 'dt' constant if present (EPOCH writes one).
    double dt = 0.0;
    bool hasDt = false;
};

BlockMap
classify(std::vector<sdf_block_t *> const & blocks);

} // namespace sdf2opmd
