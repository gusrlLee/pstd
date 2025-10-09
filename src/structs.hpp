#pragma once 
#include "types.hpp"

namespace pstd
{
    struct QueueFamilyIndices 
    {
        std::optional<uint32_t> graphicsFamily;
        bool isComplete() 
        {
            return graphicsFamily.has_value();
        }
    };
}