#define STB_IMAGE_IMPLEMENTATION
#include "rtw_stb_image.h"

rtw_image::~rtw_image() {
    STBI_FREE(data);
}