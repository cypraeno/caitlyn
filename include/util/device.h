#ifndef DEVICE_H
#define DEVICE_H
#include <iostream>

/**
 * @brief Handles error output for the rendering device.
 */
void errorFunction(void* userPtr, enum RTCError error, const char* str) {
    std::cout << "error " << error << ": " << str << std::endl;
}

/**
 * @brief Initializes and returns a new rendering device.
 * 
 * @return RTCDevice A handle to the newly created device, or nullptr if creation failed.
 */

RTCDevice initializeDevice() {
  RTCDevice device = rtcNewDevice(NULL);

  if (!device)
    std::cout << "error " << rtcGetDeviceError(NULL) << ": cannot create device" << std::endl;

  rtcSetDeviceErrorFunction(device, errorFunction, NULL);
  return device;
}

#endif