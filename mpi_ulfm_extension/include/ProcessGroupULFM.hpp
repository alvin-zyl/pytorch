#pragma once

// #ifdef USE_C10D_MPI

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <mpi.h>
#include <torch/python.h>
#include <pybind11/chrono.h>
#include "TypesULFM.hpp"

namespace c10d {

constexpr const char* ULFM_BACKEND_NAME = "mpi";

// Recovery status enum to differentiate between different outcomes
enum class RecoveryStatus {
  NO_FAILURE,        // No failure detected, system is healthy
  RECOVERED,         // Failure(s) detected and successfully recovered
  FAILED_TO_RECOVER  // Failure(s) detected but recovery failed
};

// Result type for recovery operations that carries status, error info, and failure details
struct RecoveryResult {
  RecoveryStatus status;
  std::string error_message;
  int error_code;               // MPI error code if applicable (0 if not applicable)
  int num_failed_ranks;         // Number of failed ranks detected (0 if no failure)

  // Constructors
  RecoveryResult()
      : status(RecoveryStatus::NO_FAILURE), error_code(0), num_failed_ranks(0) {}

  RecoveryResult(RecoveryStatus s, const std::string& msg = "", int code = 0, int failed = 0)
      : status(s), error_message(msg), error_code(code), num_failed_ranks(failed) {}

  // Factory methods for convenience
  static RecoveryResult NoFailure() {
    return RecoveryResult(RecoveryStatus::NO_FAILURE);
  }

  static RecoveryResult Recovered(int num_failed = 0) {
    return RecoveryResult(RecoveryStatus::RECOVERED, "", 0, num_failed);
  }

  static RecoveryResult Error(const std::string& msg, int code = 0) {
    return RecoveryResult(RecoveryStatus::FAILED_TO_RECOVER, msg, code, 0);
  }

  // Query methods
  bool is_ok() const {
    return status == RecoveryStatus::NO_FAILURE || status == RecoveryStatus::RECOVERED;
  }

  bool has_failure() const {
    return status == RecoveryStatus::RECOVERED || status == RecoveryStatus::FAILED_TO_RECOVER;
  }

  bool failed_to_recover() const {
    return status == RecoveryStatus::FAILED_TO_RECOVER;
  }

  // Implicit conversion to bool for backward compatibility (true if ok)
  operator bool() const { return is_ok(); }
};

// WorkEntry is the state associated with a single MPI run instance.
// It include the source Tensor list and destination Tensor list, as well as
// The actual run function that will operate either on src or dst or both.
struct WorkEntry {
  explicit WorkEntry(
      std::vector<at::Tensor>* srcPtr,
      std::vector<at::Tensor>* dstPtr,
      std::function<void(std::unique_ptr<WorkEntry>&)> run)
      : dst(dstPtr ? *dstPtr : std::vector<at::Tensor>()), run(std::move(run)) {
    if (srcPtr) {
      src = *srcPtr;
    }
  }

  // Not copyable
  WorkEntry(const WorkEntry&) = delete;
  // Not copy assignable
  WorkEntry& operator=(const WorkEntry&) = delete;

  // For input and output tensors (in-place), we will always use src
  std::vector<at::Tensor> src;

  // Copy of user provided outputs.
  const std::vector<at::Tensor> dst;

  // src rank returned, for recv only
  int* srcRank = nullptr;
  std::function<void(std::unique_ptr<WorkEntry>&)> run;
  
  // For ULFM work entries - pointer to WorkULFM for failure recording  
  void* ulfmWork = nullptr;
};

// ProcessGroupMPI implements MPI bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes.
//
// All MPI functions provided by this class is asynchronously scheduled on a
// Worker thread. Therefore, ProcessGroupMPI requires the MPI implementation
// that is used to have a minimum thread support value of MPI_THREAD_SERIALIZED.
// That is, The process may be multi-threaded, and multiple threads may make
// MPI calls, but only one at a time: MPI calls are not made concurrently from
// two distinct threads (all MPI calls are serialized). However, with
// MPI_THREAD_SERIALIZED, ProcessGroupMPI will only support a single process
// group. In other words, no more than 1 process group can be created globally.
//
// If you would like to use multiple ProcessGroupMPI, it requires your MPI
// implementation to have a thread support value of MPI_THREAD_MULTIPLE, that
// is, multiple threads may call MPI, with no restriction.
//
// Also note that ProcessGroupMPI only supports a single Tensor operation. In
// other words, the size of the input Tensor vector should always be 1.
//
// CUDA tensor can be supported if the MPI used is CUDA-aware MPI, and
// ProcessGroupMPI will automatically detect this support.
class TORCH_API ProcessGroupULFM : public ProcessGroup {
 public:
  class WorkMPI : public Work {
   public:
    explicit WorkMPI(
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt)
        : Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
          outputTensors_(std::move(outputTensors)),
          future_(c10::make_intrusive<at::ivalue::Future>(
              c10::ListType::create(c10::TensorType::get()))) {}

    std::vector<at::Tensor> result() override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   protected:
    friend class ProcessGroupULFM;

   private:
    void finishWorkMPI();
    void finishWorkMPIError(const std::exception_ptr& eptr);

    std::vector<at::Tensor> outputTensors_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
  };

  class WorkULFM : public WorkMPI {
   public:
    explicit WorkULFM(
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt)
        : WorkMPI(std::move(outputTensors), profilingTitle, inputTensors),
          hasFailures_(false) {}

    // Failure detection methods (no recovery)
    bool has_failures() const;
    std::vector<int> get_failed_ranks() const;
    bool was_noop() const {
      return was_noop_.load(std::memory_order_acquire);
    }
    void markNoop() {
      was_noop_.store(true, std::memory_order_release);
    }


   protected:
    friend class ProcessGroupULFM;
    void recordFailure(const std::vector<int>& failedRanks);

   private:
    mutable std::mutex failureMutex_;
    bool hasFailures_;
    std::vector<int> failedRanks_;
    std::atomic<bool> was_noop_{false};
  };

  class AsyncWork : public Work {
   public:
    AsyncWork(
        MPI_Request request,
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt);

    ~AsyncWork() override;

    bool isCompleted() override;

    bool isSuccess() const override;

    int sourceRank() const override;

    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

    void abort() override;

    std::vector<at::Tensor> result() override;

   protected:
    void populateException();

   private:
    const std::vector<at::Tensor> outputTensors_;
    MPI_Request request_;
    MPI_Status status_{};
  };

  // Constructor will spawn up the worker thread loop
  explicit ProcessGroupULFM(int rank, int size, MPI_Comm pgComm);

  ~ProcessGroupULFM() override;

  // Abort the MPI program, needs to be called when exception is detected
  void abort() override;

  const std::string getBackendName() const override {
      return std::string(ULFM_BACKEND_NAME);
    }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> ulfm_allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions(),
      const ULFMOptions& ulfm_opts = ULFMOptions()
    );

  // Recovery methods (ProcessGroup-level operations)
  RecoveryResult repair_communicator();
  void notify_all_ranks_of_failure();
  bool check_for_failures() const;

  void set_quiesce(bool v) {
    quiesce_.store(v, std::memory_order_release);
  }
  bool is_quiesced() const {
    return quiesce_.load(std::memory_order_acquire);
  }

  // Epoch (bumps after repair)
  int worldEpoch() const noexcept {
    return world_epoch_.load(std::memory_order_acquire);
  }


  // Comprehensive failure detection and recovery workflow
  RecoveryResult detect_and_recover_failures(bool auto_repair = true, std::vector<int>* failed_ranks = nullptr);

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  // Creating a new ProcessGroupMPI, will initialize MPI if not initialized
  static c10::intrusive_ptr<ProcessGroup> createProcessGroupULFM(
      std::vector<int> ranks = {});

  static void ProcessGroupULFMConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    register_backend("ulfm", py::cpp_function(createProcessGroupULFM));
  }

 protected:
  using WorkType =
      std::tuple<std::unique_ptr<WorkEntry>, c10::intrusive_ptr<WorkMPI>>;
  // Worker thread loop
  void runLoop();
  // Helper function that is called by the destructor
  void destroy();

  // Modular failure recovery helper methods (corrected workflow order)
  bool notice_failure();
  RecoveryResult get_failed_ranks_internal(std::vector<int>& failed_ranks_comm, std::vector<int>* failed_ranks_world = nullptr);
  RecoveryResult ack_failures();
  RecoveryResult agree_on_failed_ranks(const std::vector<int>& failed_ranks);
  bool should_repair_communicator(const std::vector<int>& failed_ranks);
  RecoveryResult repair_communicator_internal();
  void bumpEpoch() {
    world_epoch_.fetch_add(1, std::memory_order_acq_rel);
  }

  c10::intrusive_ptr<Work> enqueue(
      std::unique_ptr<WorkEntry> entry,
      const char* profilingTitle = nullptr,
      const std::optional<std::vector<at::Tensor>>& inputTensors =
          std::nullopt);

  c10::intrusive_ptr<WorkULFM> enqueueULFM(
      std::unique_ptr<WorkEntry> entry,
      const char* profilingTitle = nullptr,
      const std::optional<std::vector<at::Tensor>>& inputTensors =
          std::nullopt);

  bool stop_;

  std::mutex pgMutex_;
  std::thread workerThread_;

  std::deque<WorkType> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

  // Global states
  static void initMPIOnce();
  static void mpiExit();

  static std::mutex pgGlobalMutex_;
  static int mpiThreadSupport_;

  MPI_Comm pgComm_;
  int currentRank_;  // Current rank after repairs (may change)
  int currentSize_;  // Current size after repairs (may change)

 private:
  std::atomic<bool> quiesce_{false};
  std::atomic<int>  world_epoch_{0};
};

} // namespace c10d

// #endif // USE_C10D_MPI
