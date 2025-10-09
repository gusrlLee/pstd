#include "utils.hpp"

namespace pstd
{
	namespace utils
	{
		void printAvailableInstanceExtensions()
		{
			uint32_t extensionCount = 0;
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
			std::vector<VkExtensionProperties> extensions(extensionCount);
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
			
			fmt::print("[Available instance extensions]:\n");
			for (const auto& extension : extensions) 
			{
				fmt::print("\t {} \n", extension.extensionName);
			}
			fmt::print("\n");
		}

		void printAvailablPhyscialDevices(Context& ctx)
		{
			uint32_t deviceCount = 0;
			vkEnumeratePhysicalDevices(ctx.inst, &deviceCount, nullptr);
			std::vector<VkPhysicalDevice> devices(deviceCount);
			vkEnumeratePhysicalDevices(ctx.inst, &deviceCount, devices.data());
			
			fmt::print("[Available physical devices list]:\n");
			for (const auto& device : devices)
			{
				VkPhysicalDeviceProperties deviceProperties;
				vkGetPhysicalDeviceProperties(device, &deviceProperties);
				VkPhysicalDeviceFeatures deviceFeatures;
				vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
				fmt::print("\t {}, {} \n", deviceProperties.deviceID, deviceProperties.deviceName);
			}
			fmt::print("\n");
		}

		void printAvailableDeviceExtensions(Context &ctx)
		{
			uint32_t extensionCount = 0;
			VK_CHECK(vkEnumerateDeviceExtensionProperties(ctx.gpu, nullptr, &extensionCount, nullptr));
			std::vector<VkExtensionProperties> extensions(extensionCount);
			VK_CHECK(vkEnumerateDeviceExtensionProperties(ctx.gpu, nullptr, &extensionCount, extensions.data()));
			
			fmt::print("[Available logical device extensions]:\n");
			for (const auto& ext : extensions)
			{
				fmt::print("\t {} version: {} \n", ext.extensionName, ext.specVersion);
			}
		}
	}
}
