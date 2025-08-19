#include "ULFMReducer.hpp"

#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/default_comm_hooks.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/utils/pybind.h>

namespace c10d {

ULFMCommHook::ULFMCommHook(
    c10::intrusive_ptr<ProcessGroupULFM> state,
    ULFMFailureHandlingStrategy failure_strategy)
    : CppCommHookInterface<c10::intrusive_ptr<ProcessGroupULFM>>(std::move(state)),
      failure_strategy_(failure_strategy) {}

c10::intrusive_ptr<c10::ivalue::Future> ULFMCommHook::runHook(GradBucket& bucket) {
  std::lock_guard<std::mutex> lock(ulfm_mutex_);
  
  try {
    // Get the bucket tensor
    std::vector<at::Tensor> tensors = {bucket.getBuffer()};
    
    // Create allreduce options (DDP uses SUM by default)
    AllreduceOptions allreduce_opts;
    allreduce_opts.reduceOp = ReduceOp::SUM;
    
    // Create ULFM options
    ULFMOptions ulfm_opts = get_ulfm_options();
    
    // Perform ULFM allreduce using your existing implementation
    auto ulfm_pg = get_ulfm_process_group();
    auto work = ulfm_pg->ulfm_allreduce(tensors, allreduce_opts, ulfm_opts);
    
    // Mark communicator as healthy if we get here
    communicator_healthy_.store(true);
    
    // Create a future that will return the tensor when work completes
    auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
    
    // Set up completion callback
    work->getFuture()->addCallback([fut, tensor = bucket.getBuffer()](c10::ivalue::Future& work_fut) mutable {
      try {
        // Wait for work completion
        work_fut.wait();
        
        // Return the tensor (it was modified in-place by allreduce)
        fut->markCompleted(c10::IValue(tensor));
      } catch (const std::exception& e) {
        fut->setError(std::make_exception_ptr(e));
      }
    });
    
    return fut;
    
  } catch (const std::exception& e) {
    // Mark communicator as potentially unhealthy
    communicator_healthy_.store(false);
    
    // Log the error
    TORCH_WARN("ULFM allreduce failed: ", e.what());
    
    // Handle the failure based on strategy
    if (failure_strategy_ == ULFMFailureHandlingStrategy::ABORT_ON_FAILURE) {
      TORCH_CHECK(false, "Aborting due to process failure in ULFM allreduce: ", e.what());
    }
    
    // For CONTINUE_WITH_SURVIVORS, return the original tensor
    auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
    fut->markCompleted(c10::IValue(bucket.getBuffer()));
    
    return fut;
  }
}

void ULFMCommHook::set_failure_handling_strategy(ULFMFailureHandlingStrategy strategy) {
  std::lock_guard<std::mutex> lock(ulfm_mutex_);
  failure_strategy_ = strategy;
}

ULFMFailureHandlingStrategy ULFMCommHook::get_failure_handling_strategy() const {
  std::lock_guard<std::mutex> lock(ulfm_mutex_);
  return failure_strategy_;
}

bool ULFMCommHook::is_communicator_healthy() const {
  return communicator_healthy_.load();
}

bool ULFMCommHook::repair_communicator() {
  std::lock_guard<std::mutex> lock(ulfm_mutex_);
  
  try {
    // Since your ulfm_allreduce automatically handles repair with auto_repair=true,
    // we just mark it as healthy and let the next collective operation handle repair
    communicator_healthy_.store(true);
    TORCH_WARN("Communicator marked as healthy - repair will happen in next collective operation");
    return true;
  } catch (const std::exception& e) {
    TORCH_WARN("Failed to mark communicator as healthy: ", e.what());
    return false;
  }
}

ULFMOptions ULFMCommHook::get_ulfm_options() const {
  ULFMOptions ulfm_opts;
  ulfm_opts.auto_repair = true;  // Always enable auto-repair
  ulfm_opts.failure_strategy = failure_strategy_;
  ulfm_opts.max_retries = 3;
  ulfm_opts.retry_delay_ms = 100;
  return ulfm_opts;
}

c10::intrusive_ptr<ProcessGroupULFM> ULFMCommHook::get_ulfm_process_group() const {
  // state_ is c10::intrusive_ptr<ProcessGroupULFM>
  return state_;
}

// Function to create ULFM communication hook
std::unique_ptr<ULFMCommHook> create_ulfm_hook(
    c10::intrusive_ptr<ProcessGroupULFM> process_group,
    ULFMFailureHandlingStrategy failure_strategy) {
  return std::make_unique<ULFMCommHook>(process_group, failure_strategy);
}

} // namespace c10d
