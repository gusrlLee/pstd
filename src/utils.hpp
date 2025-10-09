#pragma once 
#include "types.hpp"
#include "Context.hpp"

namespace pstd
{
	namespace utils
	{
		void printAvailableInstanceExtensions();
		void printAvailablPhyscialDevices(Context& ctx);
		void printAvailableDeviceExtensions(Context& ctx);
	}
}