#include "pstd.hpp"

int main()
{
	pstd::utils::printAvailableInstanceExtensions();
	pstd::ContextCreateInfo ctxInfo;
	ctxInfo.useValidationLayer = true;

	// for tensor core
	ctxInfo.devExts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

	// for rt core 
	ctxInfo.devExts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_KHR_SHADER_SUBGROUP_EXTENDED_TYPES_EXTENSION_NAME);
	ctxInfo.devExts.push_back(VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME);
	
	pstd::Context ctx = pstd::createContext(ctxInfo);
	pstd::utils::printAvailablPhyscialDevices(ctx);
	pstd::utils::printAvailableDeviceExtensions(ctx);

	pstd::destroyContext(ctx);
	return 0;
}