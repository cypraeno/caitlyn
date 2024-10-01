#ifndef DEVICE_H
#define DEVICE_H
#include <iostream>
#include <embree4/rtcore.h>

/**
 * @brief Handles error output for the rendering device.
 */
void errorFunction(void* userPtr, enum RTCError error, const char* str);

/**
 * @brief Initializes and returns a new rendering device.
 * 
 * @return RTCDevice A handle to the newly created device, or nullptr if creation failed.
 */
RTCDevice initializeDevice();

#endif
