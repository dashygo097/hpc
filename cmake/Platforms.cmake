# Platform detection and platform-specific defaults

if(APPLE)
  message(STATUS "Building on macOS")
  
elseif(UNIX)
  message(STATUS "Building on Unix-like system")
  
elseif(WIN32)
  message(STATUS "Building on Windows")
  
else()
  message(FATAL_ERROR "Unsupported platform")
endif()

# Create platform config string for config.cmake. in
set(PLATFORM_CONFIG "# Platform: ${CMAKE_SYSTEM_NAME}")
