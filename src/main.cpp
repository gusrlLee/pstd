#include "pstd.hpp"

int main()
{
	pstd::utils::printAvailableInstanceExtensions();
	pstd::ContextCreateInfo ctxInfo;
	ctxInfo.useValidationLayer = false;

	// for swapchain
	ctxInfo.devExts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

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

	pstd::QueueFamilyIndices indices = pstd::findQueueFamilies(ctx.gpu);
	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = indices.graphicsFamily.value();
	VkCommandPool commandPool;
	VK_CHECK(vkCreateCommandPool(ctx.dev, &poolInfo, NULL, &commandPool));

	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	VK_CHECK(vkAllocateCommandBuffers(ctx.dev, &allocInfo, &commandBuffer));

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
	/*
		Somethings
	*/
	VK_CHECK(vkEndCommandBuffer(commandBuffer));

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	VK_CHECK(vkQueueSubmit(ctx.gq, 1, &submitInfo, VK_NULL_HANDLE));

	vkQueueWaitIdle(ctx.gq);
	printf("Command buffer submitted and queue waited idle.\n");

	vkDestroyCommandPool(ctx.dev, commandPool, nullptr);
	pstd::destroyContext(ctx);
	return 0;
}