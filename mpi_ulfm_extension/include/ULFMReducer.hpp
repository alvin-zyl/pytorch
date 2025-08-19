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

// Function to create ULFM communication hook
TORCH_API std::unique_ptr<ULFMCommHook> create_ulfm_hook(
    c10::intrusive_ptr<ProcessGroupULFM> process_group,
    ULFMFailureHandlingStrategy failure_strategy = ULFMFailureHandlingStrategy::CONTINUE_WITH_SURVIVORS);

} // namespace c10d