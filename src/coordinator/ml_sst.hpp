#pragma once

#include <vector>

#include "sst.h"

namespace sst {
class MLSST : public SST<MLSST> {
public:
    MLSST(const std::vector<uint32_t>& members, const uint32_t my_id,
          const size_t num_params, const uint32_t num_nodes);

    SSTFieldVector<double> model_or_gradient;
    SSTField<uint64_t> round;
    SSTFieldVector<uint64_t> last_round;

private:
    void initialize();
};
}  // namespace sst
