#include "device.h"

void errorFunction(void* userPtr, enum RTCError error, const char* str) {
    std::cout << "error " << error << ": " << str << std::endl;
}

RTCDevice initializeDevice() {
  RTCDevice device = rtcNewDevice(NULL);

  if (!device)
    std::cout << "error " << rtcGetDeviceError(NULL) << ": cannot create device" << std::endl;

  rtcSetDeviceErrorFunction(device, errorFunction, NULL);
  return device;
}
