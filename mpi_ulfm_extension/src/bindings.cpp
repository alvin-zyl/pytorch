#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ProcessGroupULFM.hpp"
#include "TypesULFM.hpp"
#include "ULFMLogging.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupULFM", &c10d::ProcessGroupULFM::createProcessGroupULFM);

  py::class_<c10d::ULFMOptions>(m, "ULFMOptions")
      .def(py::init<>())
      .def(py::init([](bool auto_repair) {
          c10d::ULFMOptions opts;
          opts.auto_repair = auto_repair;
          return opts;
      }),
      py::arg("auto_repair") = false)
      .def_readwrite("auto_repair", &c10d::ULFMOptions::auto_repair)
      .def_readwrite("failure_strategy", &c10d::ULFMOptions::failure_strategy);

  py::enum_<c10d::ULFMFailureHandlingStrategy>(m, "ULFMFailureHandlingStrategy")
      .value("CONTINUE_WITH_SURVIVORS", c10d::ULFMFailureHandlingStrategy::CONTINUE_WITH_SURVIVORS)
      .value("RESTART_FAILED_PROCESSES", c10d::ULFMFailureHandlingStrategy::RESTART_FAILED_PROCESSES)
      .value("ABORT_ON_FAILURE", c10d::ULFMFailureHandlingStrategy::ABORT_ON_FAILURE);

  py::class_<c10d::ProcessGroupULFM, c10d::ProcessGroup, c10::intrusive_ptr<c10d::ProcessGroupULFM>>(m, "ProcessGroupULFM")
      .def("ulfm_allreduce", &c10d::ProcessGroupULFM::ulfm_allreduce,
      py::arg("tensors"),
      py::arg("opts") = c10d::AllreduceOptions(),
      py::arg("ulfm_opts") = c10d::ULFMOptions())
      .def("repair_communicator", &c10d::ProcessGroupULFM::repair_communicator)
      .def("notify_all_ranks_of_failure", &c10d::ProcessGroupULFM::notify_all_ranks_of_failure)
      .def("check_for_failures", &c10d::ProcessGroupULFM::check_for_failures)
      .def("detect_and_recover_failures", [](c10d::ProcessGroupULFM& self, bool auto_repair) {
          std::vector<int> failed_ranks;
          bool success = self.detect_and_recover_failures(auto_repair, &failed_ranks);
          return py::make_tuple(success, failed_ranks);
      }, py::arg("auto_repair") = true,
         "Comprehensive failure detection and recovery workflow. Returns (success, failed_ranks)")
      .def("consensus", &c10d::ProcessGroupULFM::consensus,
         py::arg("ulfm_opts") = c10d::ULFMOptions(),
         "Perform consensus operation for failure detection and recovery")
      .def("set_quiesce", &c10d::ProcessGroupULFM::set_quiesce, py::arg("v"),
         "Set the quiesce state of the process group")
      .def("is_quiesced", &c10d::ProcessGroupULFM::is_quiesced,
         "Check if the process group is currently quiesced")
      .def("worldEpoch", &c10d::ProcessGroupULFM::worldEpoch,
         "Get the current world epoch (increments after communicator repairs)")
      .def("current_rank", &c10d::ProcessGroupULFM::current_rank,
         "Get the current rank (may change after communicator repairs)")
      .def("current_size", &c10d::ProcessGroupULFM::current_size,
         "Get the current world size (may change after communicator repairs)");

  py::class_<c10d::ProcessGroupULFM::WorkULFM, c10d::Work, c10::intrusive_ptr<c10d::ProcessGroupULFM::WorkULFM>>(m, "WorkULFM")
      .def("has_failures", &c10d::ProcessGroupULFM::WorkULFM::has_failures)
      .def("get_failed_ranks", &c10d::ProcessGroupULFM::WorkULFM::get_failed_ranks)
      .def("was_noop", &c10d::ProcessGroupULFM::WorkULFM::was_noop,
         "Check if this work was marked as a no-op due to failures")
      .def("markNoop", &c10d::ProcessGroupULFM::WorkULFM::markNoop,
         "Mark this work as a no-op (used internally for failure handling)");

  // ULFM logging control
  m.def("set_ulfm_verbose_logging", &c10d::set_ulfm_verbose_logging, 
        py::arg("verbose"), 
        "Enable or disable verbose ULFM logging");
  
  m.def("is_ulfm_verbose_logging", &c10d::is_ulfm_verbose_logging, 
        "Check if verbose ULFM logging is enabled");
}
