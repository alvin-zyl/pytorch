#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ProcessGroupULFM.hpp"
#include "TypesULFM.hpp"
#include "ULFMReducer.hpp"
#include "ULFMLogging.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupULFM", &c10d::ProcessGroupULFM::createProcessGroupULFM);

  py::class_<c10d::ULFMOptions>(m, "ULFMOptions")
      .def(py::init<>())
      .def_readwrite("auto_repair", &c10d::ULFMOptions::auto_repair)
      .def_readwrite("failure_strategy", &c10d::ULFMOptions::failure_strategy)
      .def_readwrite("max_retries", &c10d::ULFMOptions::max_retries)
      .def_readwrite("retry_delay_ms", &c10d::ULFMOptions::retry_delay_ms);

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
         "Comprehensive failure detection and recovery workflow. Returns (success, failed_ranks)");

  py::class_<c10d::ProcessGroupULFM::WorkULFM, c10d::Work, c10::intrusive_ptr<c10d::ProcessGroupULFM::WorkULFM>>(m, "WorkULFM")
      .def("has_failures", &c10d::ProcessGroupULFM::WorkULFM::has_failures)
      .def("get_failed_ranks", &c10d::ProcessGroupULFM::WorkULFM::get_failed_ranks);

  py::class_<c10d::ULFMCommHook>(m, "ULFMCommHook")
      .def(py::init<
          c10::intrusive_ptr<c10d::ProcessGroupULFM>,
          c10d::ULFMFailureHandlingStrategy>(),
          py::arg("process_group"),
          py::arg("failure_strategy") = c10d::ULFMFailureHandlingStrategy::CONTINUE_WITH_SURVIVORS)
      .def("set_failure_handling_strategy", &c10d::ULFMCommHook::set_failure_handling_strategy,
          py::arg("strategy"))
      .def("get_failure_handling_strategy", &c10d::ULFMCommHook::get_failure_handling_strategy)
      .def("is_communicator_healthy", &c10d::ULFMCommHook::is_communicator_healthy)
      .def("repair_communicator", &c10d::ULFMCommHook::repair_communicator);

  // Note: Reducer class is already bound by PyTorch

  m.def("create_ulfm_hook", [](
      c10::intrusive_ptr<c10d::ProcessGroupULFM> process_group,
      c10d::ULFMFailureHandlingStrategy failure_strategy = c10d::ULFMFailureHandlingStrategy::CONTINUE_WITH_SURVIVORS
  ) {      
      return c10d::create_ulfm_hook(process_group, failure_strategy);
  },
      py::arg("process_group"),
      py::arg("failure_strategy") = c10d::ULFMFailureHandlingStrategy::CONTINUE_WITH_SURVIVORS,
      "Create ULFM communication hook for use with PyTorch DDP",
      py::return_value_policy::automatic);

  // ULFM logging control
  m.def("set_ulfm_verbose_logging", &c10d::set_ulfm_verbose_logging, 
        py::arg("verbose"), 
        "Enable or disable verbose ULFM logging");
  
  m.def("is_ulfm_verbose_logging", &c10d::is_ulfm_verbose_logging, 
        "Check if verbose ULFM logging is enabled");
}
