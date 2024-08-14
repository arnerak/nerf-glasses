#include "common.h"

namespace common {
    void ensureMinimumSize(uint32_t& width, uint32_t& height) {
        if(width == 0) {
            width = 1;
        }
        if(height == 0) {
            height = 1;
        }
    }
}