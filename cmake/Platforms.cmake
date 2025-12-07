# Platform detection and platform-specific defaults

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  message(STATUS "Building on macOS")
  message(STATUS "Detected Apple Silicon (${CMAKE_SYSTEM_PROCESSOR})")

elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i386)$")
  message(STATUS "Building on macOS")
  message(STATUS "Detected Intel architecture (${CMAKE_SYSTEM_PROCESSOR})")
  
elseif(UNIX)
  message(STATUS "Building on Unix-like system")
  
elseif(WIN32)
  message(STATUS "Building on Windows")
  
else()
  message(FATAL_ERROR "Unsupported platform")
endif()

# Create platform config string for config.cmake. in
set(PLATFORM_CONFIG "# Platform: ${CMAKE_SYSTEM_NAME}")
