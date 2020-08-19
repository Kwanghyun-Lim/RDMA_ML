#include "ml_sst.hpp"
#include "utils/utils.hpp"


sst::MLSST::MLSST(const std::vector<uint32_t>& members,
                  const uint32_t my_id, const size_t num_params, const uint32_t num_nodes)
        : sst::SST<MLSST>(this, SSTParams{members, my_id}),
          model_or_gradient(num_params),
	  last_round(num_nodes) {
    SSTInit(model_or_gradient, round, last_round);
    initialize();
}

void sst::MLSST::initialize() {
    uint32_t num_nodes = get_num_rows();
    for(uint i = 0; i < num_nodes; ++i) {
        utils::zero_arr((double*)std::addressof(
                                model_or_gradient[i][0]),
                        model_or_gradient.size());
    }
    for(uint i = 0; i < num_nodes; ++i) {
        round[i] = 0;
    }
    
    for(uint i = 0; i < num_nodes; ++i) {
      for(uint j = 0; j < last_round.size(); ++j) {
        last_round[i][j] = 0;
      }
    }
    put_with_completion();  // end of initialization
    sync_with_members();
}

void sst::MLSST::print() {
    uint32_t num_nodes = get_num_rows();

    std::cout << "round ";
    for(uint i = 0; i < num_nodes; ++i) {
      std::cout << round[i] << " ";
    }
    std::cout << std::endl;

    // std::cout << "relay_to";
    // std::cout << std::endl;
    // for(uint i = 0; i < num_nodes; ++i) {
    //   std::cout << "[" << i << "] ";
    //   for(uint j = 0; j < relay_to.size(); ++j) {
    // 	std::cout << relay_to[i][j] << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << std::endl;

    std::cout << "last_round";
    std::cout << std::endl;
    for(uint i = 0; i < num_nodes; ++i) {
      std::cout << "[" << i << "] ";
      for(uint j = 0; j < last_round.size(); ++j) {
	std::cout << last_round[i][j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
}

// sst::MLSST::MLSST(const std::vector<uint32_t>& members,
//                   const uint32_t my_id, const size_t num_params, const uint32_t num_nodes)
//         : sst::SST<MLSST>(this, SSTParams{members, my_id}),
//           model_or_gradient(num_params),
//           relay_to(num_nodes),
// 	  relay_model(num_params),
// 	  last_round(num_nodes),
//           model_or_gradient1(num_params),
//           relay_to1(num_nodes),
// 	  relay_model1(num_params),
// 	  last_round1(num_nodes),
//           model_or_gradient2(num_params),
//           relay_to2(num_nodes),
// 	  relay_model2(num_params),
// 	  last_round2(num_nodes) {
//   SSTInit(model_or_gradient, round, relay_to, relay_model, last_round,
// 	  model_or_gradient1, round1, relay_to1, relay_model1, last_round1,
// 	  model_or_gradient2, round2, relay_to2, relay_model2, last_round2);
//   initialize();
// }

// void sst::MLSST::initialize() {
//     uint32_t num_nodes = get_num_rows();
//     for(uint i = 0; i < num_nodes; ++i) {
//         utils::zero_arr((double*)std::addressof(
//                                 model_or_gradient[i][0]),
//                         model_or_gradient.size());
// 	utils::zero_arr((double*)std::addressof(
// 						model_or_gradient1[i][0]),
// 			model_or_gradient1.size());
// 	utils::zero_arr((double*)std::addressof(
// 						model_or_gradient2[i][0]),
// 			model_or_gradient2.size());
//     }
    
//     for(uint i = 0; i < num_nodes; ++i) {
//         round[i] = 0;
// 	round1[i] = 0;
//         round2[i] = 0;
//     }

//     for(uint i = 0; i < num_nodes; ++i) {
//       for(uint j = 0; j < relay_to.size(); ++j) {
//         relay_to[i][j] = -1;
//         relay_to1[i][j] = -1;
//         relay_to2[i][j] = -1;
//       }
//     }

//     for(uint i = 0; i < num_nodes; ++i) {
//         utils::zero_arr((double*)std::addressof(relay_model[i][0]), relay_model.size());
//         utils::zero_arr((double*)std::addressof(relay_model1[i][0]), relay_model1.size());
//         utils::zero_arr((double*)std::addressof(relay_model2[i][0]), relay_model2.size());
//     }
    
//     for(uint i = 0; i < num_nodes; ++i) {
//       for(uint j = 0; j < last_round.size(); ++j) {
//         last_round[i][j] = 0;
// 	last_round1[i][j] = 0;
//         last_round2[i][j] = 0;
//       }
//     }
    
//     put_with_completion();  // end of initialization
//     sync_with_members();
// }
