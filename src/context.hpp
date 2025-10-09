#pragma once 

#include "types.hpp"
#include "structs.hpp"

namespace pstd
{
	struct ContextCreateInfo
	{
		bool useValidationLayer = false;
		std::vector<const char*> layers;
		std::vector<const char*> exts;
		std::vector<const char*> devExts;
	};

	struct Context
	{
		VkInstance inst = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT dm = VK_NULL_HANDLE;
		VkPhysicalDevice gpu = VK_NULL_HANDLE;
		VkDevice dev = VK_NULL_HANDLE;

		VkQueue gq;
	};

	Context createContext(ContextCreateInfo& info);
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
	void destroyContext(Context& ctx);
}