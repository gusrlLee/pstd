#include "context.hpp"
#include <iostream>
#include "GLFW/glfw3.h"

namespace pstd
{
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		std::cerr << "[VALIDATION LAYER]: " << pCallbackData->pMessage << std::endl;
		return VK_FALSE;
	}

	VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
	{
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
		if (func != nullptr)
		{
			return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
		}
		else
		{
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}

	void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
	{
		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
		if (func != nullptr)
		{
			func(instance, debugMessenger, pAllocator);
		}
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
	{
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	bool checkValidationLayerSupport(ContextCreateInfo& info)
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : info.layers)
		{
			bool layerFound = false;
			for (const auto& layerProperties : availableLayers)
			{
				if (strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
			{
				return false;
			}
		}

		return true;
	}

	std::vector<const char*> getRequiredGlfwExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		return extensions;
	}


	/////////////////////////////////////////////////////////////////////////////
	// INSTANCE FUNCTION
	/////////////////////////////////////////////////////////////////////////////

	void createInstance(Context& ctx, ContextCreateInfo& info)
	{
		if (info.useValidationLayer)
		{
			info.layers.push_back("VK_LAYER_KHRONOS_validation");
			if (!checkValidationLayerSupport(info))
				throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "pstd app";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "pstd engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_3;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		std::vector<const char*> glfwExtensions = getRequiredGlfwExtensions();
		info.exts.insert(info.exts.begin(), glfwExtensions.begin(), glfwExtensions.end());
		if (info.useValidationLayer)
		{
			info.exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		createInfo.enabledExtensionCount = info.exts.size();
		createInfo.ppEnabledExtensionNames = info.exts.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (info.useValidationLayer)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(info.layers.size());
			createInfo.ppEnabledLayerNames = info.layers.data();
			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}

		VK_CHECK(vkCreateInstance(&createInfo, nullptr, &ctx.inst));
	}

	/////////////////////////////////////////////////////////////////////////////
	// DEBUG MESSENGER FUNCTION
	/////////////////////////////////////////////////////////////////////////////

	void setupDebugMessenger(Context& ctx, ContextCreateInfo& info)
	{
		if (!info.useValidationLayer) return;
		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);
		VK_CHECK(CreateDebugUtilsMessengerEXT(ctx.inst, &createInfo, nullptr, &ctx.dm));
	}

	/////////////////////////////////////////////////////////////////////////////
	// PHYSICAL DEVICE FUNCTION
	/////////////////////////////////////////////////////////////////////////////

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
		int i = 0;
		for (const auto& queueFamily : queueFamilies)
		{
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphicsFamily = i;
				if (indices.isComplete())
				{
					break;
				}
			}

			i++;
		}
		return indices;
	}

	bool isDeviceSuitable(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices = findQueueFamilies(device);
		return indices.isComplete();
	}

	void pickPhysicalDevice(Context& ctx, ContextCreateInfo& info)
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(ctx.inst, &deviceCount, nullptr);
		if (deviceCount == 0)
		{
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(ctx.inst, &deviceCount, devices.data());

		for (const auto& device : devices)
		{
			if (isDeviceSuitable(device))
			{
				ctx.gpu = device;
				break;
			}
		}

		if (ctx.gpu == VK_NULL_HANDLE)
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}

	}

	/////////////////////////////////////////////////////////////////////////////
	// LOGICAL DEVICE FUNCTION
	/////////////////////////////////////////////////////////////////////////////

	bool isExtensionSupported(VkPhysicalDevice gpu, const char* name)
	{
		uint32_t extensionCount = 0;
		vkEnumerateDeviceExtensionProperties(gpu, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(gpu, nullptr, &extensionCount, extensions.data());

		for (const auto& ext : extensions)
		{
			if (strcmp(ext.extensionName, name) == 0)
				return true;
		}
		return false;
	}

	void createLogicalDevice(Context& ctx, ContextCreateInfo& info)
	{
		QueueFamilyIndices indices = findQueueFamilies(ctx.gpu);

		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
		queueCreateInfo.queueCount = 1;

		float queuePriority = 1.0f;
		queueCreateInfo.pQueuePriorities = &queuePriority;

		// ------------------------------
		// Feature Chain
		// ------------------------------
		VkPhysicalDeviceFeatures2 deviceFeatures2{};
		deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

		// (1) Ray tracing feature structs
		VkPhysicalDeviceAccelerationStructureFeaturesKHR accelStructFeatures{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
		accelStructFeatures.accelerationStructure = VK_TRUE;

		VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeatures{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
		rayTracingFeatures.rayTracingPipeline = VK_TRUE;

		VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferAddressFeatures{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR };
		bufferAddressFeatures.bufferDeviceAddress = VK_TRUE;

		// (2) Subgroup (wave intrinsic) feature structs
		VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeControlFeatures{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT };
		subgroupSizeControlFeatures.subgroupSizeControl = VK_TRUE;
		subgroupSizeControlFeatures.computeFullSubgroups = VK_TRUE;

		VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR subgroupExtendedTypesFeatures{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR };
		subgroupExtendedTypesFeatures.shaderSubgroupExtendedTypes = VK_TRUE;

		VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR subgroupUniformCF{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR };
		subgroupUniformCF.shaderSubgroupUniformControlFlow = VK_TRUE;

		// (3) Cooperative Matrix feature struct
		VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
		cooperativeMatrixFeatures.cooperativeMatrix = VK_TRUE;

		// ------------------------------
		// pNext Connect
		// ------------------------------
		deviceFeatures2.pNext = &accelStructFeatures;
		accelStructFeatures.pNext = &rayTracingFeatures;
		rayTracingFeatures.pNext = &bufferAddressFeatures;
		bufferAddressFeatures.pNext = &subgroupSizeControlFeatures;
		subgroupSizeControlFeatures.pNext = &subgroupExtendedTypesFeatures;
		subgroupExtendedTypesFeatures.pNext = &subgroupUniformCF;
		subgroupUniformCF.pNext = &cooperativeMatrixFeatures;

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = &queueCreateInfo;
		createInfo.queueCreateInfoCount = 1;
		createInfo.enabledExtensionCount = info.devExts.size();
		createInfo.ppEnabledExtensionNames = info.devExts.data();
		createInfo.pNext = &deviceFeatures2;

		VK_CHECK(vkCreateDevice(ctx.gpu, &createInfo, nullptr, &ctx.dev));

		vkGetDeviceQueue(ctx.dev, indices.graphicsFamily.value(), 0, &ctx.gq);
	}

	/////////////////////////////////////////////////////////////////////////////
	// USER FUNCTION
	/////////////////////////////////////////////////////////////////////////////

	Context createContext(ContextCreateInfo& info)
	{
		Context ctx;
		createInstance(ctx, info);
		setupDebugMessenger(ctx, info);
		pickPhysicalDevice(ctx, info);
		createLogicalDevice(ctx, info);
		return ctx;
	}

	void destroyContext(Context& ctx)
	{
		vkDeviceWaitIdle(ctx.dev);
		vkDestroyDevice(ctx.dev, nullptr);

		if (ctx.dm != VK_NULL_HANDLE)
			DestroyDebugUtilsMessengerEXT(ctx.inst, ctx.dm, nullptr);

		vkDestroyInstance(ctx.inst, nullptr);
	}
}