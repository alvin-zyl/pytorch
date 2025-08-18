#pragma once

namespace c10d {

// Failure handling strategies for ULFM
enum class ULFMFailureHandlingStrategy {
  CONTINUE_WITH_SURVIVORS,  // Continue training with remaining processes
  RESTART_FAILED_PROCESSES, // Restart failed processes (if supported by scheduler)
  ABORT_ON_FAILURE         // Abort training on any process failure
};

struct ULFMOptions {
  bool auto_repair = false;
  ULFMFailureHandlingStrategy failure_strategy = ULFMFailureHandlingStrategy::CONTINUE_WITH_SURVIVORS;
  int max_retries = 3;
  int retry_delay_ms = 100;
  // Add more in future, like timeout, etc.
};

}
