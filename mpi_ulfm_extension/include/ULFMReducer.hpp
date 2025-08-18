#pragma once

#include <torch/csrc/distributed/c10d/reducer.hpp>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include "ProcessGroupULFM.hpp"
#include "TypesULFM.hpp"

namespace c10d {

// ULFM Communication Hook that replaces standard allreduce with ULFM allreduce
class ULFMCommHook : public CppCommHookInterface<c10::intrusive_ptr<ProcessGroupULFM>> {
 public:
  explicit ULFMCommHook(
      c10::intrusive_ptr<ProcessGroupULFM> state,
      ULFMFailureHandlingStrategy failure_strategy = ULFMFailureHandlingStrategy::CONTINUE_WITH_SURVIVORS);

  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;

  // ULFM-specific methods
  void set_failure_handling_strategy(ULFMFailureHandlingStrategy strategy);
  ULFMFailureHandlingStrategy get_failure_handling_strategy() const;
  bool is_communicator_healthy() const;
  bool repair_communicator();

 private:
  ULFMFailureHandlingStrategy failure_strategy_;
  mutable std::atomic<bool> communicator_healthy_{true};
  mutable std::mutex ulfm_mutex_;
  
  ULFMOptions get_ulfm_options() const;
  
  // Helper to get the ULFM process group from state_
  c10::intrusive_ptr<ProcessGroupULFM> get_ulfm_process_group() const;
};

// Convenience function to create a Reducer with ULFM hook
TORCH_API std::shared_ptr<Reducer> create_ulfm_reducer(
    std::vector<at::Tensor> params,
    std::vector<std::vector<size_t>> bucket_indices,
    c10::intrusive_ptr<ProcessGroupULFM> process_group,
    std::vector<bool> expect_sparse_gradients,
    int64_t bucket_bytes_cap,
    bool find_unused_parameters,
    bool gradient_as_bucket_view,
    std::unordered_map<size_t, std::string> param_names,
    int64_t first_bucket_bytes_cap,
    bool skip_all_reduce_unused_params,
    bool use_python_reducer = false,
    ULFMFailureHandlingStrategy failure_strategy = ULFMFailureHandlingStrategy::CONTINUE_WITH_SURVIVORS);

} // namespace c10d