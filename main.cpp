#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VKCPP_ENHANCED_MODE
#include <vulkan/vk_cpp.h>

#include <iostream>


vk::Format format;

vk::Instance create_instance()
{
	vk::ApplicationInfo appInfo("Test app", 1, "Engine", 1, VK_MAKE_VERSION(1, 0, 0));
	
	vk::InstanceCreateInfo instInfo;
	
	instInfo.pApplicationInfo(&appInfo);
	instInfo.flags(vk::InstanceCreateFlags());
	instInfo.enabledLayerCount(0);
	instInfo.ppEnabledLayerNames(nullptr);
	instInfo.enabledExtensionCount(1);
	const char* extensionNames[] = { "VK_KHR_xcb_surface" };
	instInfo.ppEnabledExtensionNames(extensionNames);
	
	
	return vk::createInstance(instInfo, vk::AllocationCallbacks::null());
}

vk::SurfaceKHR create_surface(const vk::Instance& inst, GLFWwindow* window)
{
	
	VkSurfaceKHR c_surface;
	auto result = glfwCreateWindowSurface(inst, window, nullptr, &c_surface);
	
	if(result != VK_SUCCESS)
	{
		std::cerr << "Error creating surface with error: " << vk::to_string(vk::Result(result));
		exit(-1);
	}
	
	return {c_surface};
	
}

vk::PhysicalDevice get_physical_device(const vk::Instance& inst)
{
	std::vector<vk::PhysicalDevice> phyDevices;
	inst.enumeratePhysicalDevices(phyDevices);
	
	// just select the first one
	if(phyDevices.empty()) throw std::runtime_error("No suitable device found");
	
	return phyDevices[0];
	
}

vk::Device create_device(const vk::Instance& inst, const vk::PhysicalDevice& physical_device)
{
	

	auto physDevProps = physical_device.getProperties();
	auto physDevQueueProps = physical_device.getQueueFamilyProperties();
	
	vk::DeviceQueueCreateInfo queueInfo;
	queueInfo.queueFamilyIndex(0);
	queueInfo.queueCount(physDevQueueProps[0].queueCount());
	
	float queue_priorities[] = {0.0f};
	queueInfo.pQueuePriorities(queue_priorities);
	
	
	vk::DeviceCreateInfo devInfo;
	devInfo.queueCreateInfoCount(1);
	devInfo.pQueueCreateInfos(&queueInfo);
	devInfo.enabledLayerCount(0);
	devInfo.ppEnabledLayerNames(nullptr);
	const char* deviceExtensionNames[1024];
	devInfo.enabledExtensionCount(0);
	devInfo.ppEnabledExtensionNames(deviceExtensionNames);
	devInfo.pEnabledFeatures(nullptr);
	
	return physical_device.createDevice(devInfo, vk::AllocationCallbacks::null());
	
}

vk::SwapchainKHR create_swapchain(const vk::Device& device, const vk::SurfaceKHR& surface, const vk::PhysicalDevice& physDev)
{
	vk::SurfaceCapabilitiesKHR capabilities;
	physDev.getSurfaceCapabilitiesKHR(surface, &capabilities);
	
	std::vector<vk::SurfaceFormatKHR> formats;
	physDev.getSurfaceFormatsKHR(surface, formats);
	
	format = formats[0].format();
	
	vk::SwapchainCreateInfoKHR swapchainInfo(vk::SwapchainCreateFlagsKHR(), surface, 2, format,
											 formats[0].colorSpace(), capabilities.currentExtent(), 1, 
											 vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment), vk::SharingMode::eExclusive, 0, nullptr, 
											 vk::SurfaceTransformFlagBitsKHR::eIdentity, vk::CompositeAlphaFlagBitsKHR::eOpaque, 
											 vk::PresentModeKHR::eFifoKHR, true, vk::SwapchainKHR());

	vk::SwapchainKHR ret;
	device.createSwapchainKHR(&swapchainInfo, nullptr, &ret);
	
	return ret;
}

vk::Semaphore create_semaphore(const vk::Device& device)
{
	vk::SemaphoreCreateInfo semaphoreInfo(vk::SemaphoreCreateFlags{});
	
	return device.createSemaphore(semaphoreInfo, vk::AllocationCallbacks::null());
}

vk::Fence create_fence(const vk::Device& device)
{
	vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlags{});
	
	return device.createFence(fenceInfo, vk::AllocationCallbacks::null());
}

vk::ImageView create_image_view(const vk::Device& device, const vk::Image& image)
{
	vk::ImageViewCreateInfo imageViewInfo(vk::ImageViewCreateFlags{}, image, vk::ImageViewType::e2D, format, 
										  vk::ComponentMapping(), vk::ImageSubresourceRange(vk::ImageAspectFlags(), 0, 1, 0, 1));
	
	return device.createImageView(imageViewInfo, vk::AllocationCallbacks::null());
}

int main()
{
	glfwSetErrorCallback([](int error, const char* desc)
	{
		std::cerr << "GLFW error " << error << ": " << desc << std::endl; 
	});
	
	glfwInit();
	
	if(glfwVulkanSupported() == GLFW_FALSE)
	{
		std::cout << "ERROR";
	}
	
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	
	GLFWwindow* window = glfwCreateWindow(1024, 720, "Here", nullptr, nullptr);

	auto inst = create_instance(); assert(inst);
	auto surface = create_surface(inst, window); assert(surface);
	auto physical_device = get_physical_device(inst); assert(physical_device);
	auto device = create_device(inst, physical_device); assert(device);
	auto swapchain = create_swapchain(device, surface, physical_device); assert(swapchain);
	
	
	std::vector<vk::Image> images;
	device.getSwapchainImagesKHR(swapchain, images);
	
	auto semaphore = create_semaphore(device);
	auto fence = create_fence(device);
	
	auto image_view = create_image_view(device, images[0]);
	
	
	vk::BufferCreateInfo buffInfo(vk::BufferCreateFlags(), sizeof(float) * 9, 
								  vk::BufferUsageFlags(vk::BufferUsageFlagBits::eVertexBuffer), vk::SharingMode::eExclusive, 0, nullptr);
	vk::Buffer buff = device.createBuffer(buffInfo, vk::AllocationCallbacks::null());
	
	
	vk::MemoryAllocateInfo allocInfo(sizeof(float) * 9, 0); // TODO: look at heap types
	auto devMemory = device.allocateMemory(allocInfo, vk::AllocationCallbacks::null());
	
	// write to the buffer
	auto mappedMemory = device.mapMemory(devMemory, 0, sizeof(float) * 9, vk::MemoryMapFlags());
	
	float triData[] = {
		-1.f, -1.f, 0.f,
		 1.f, -1.f, 0.f,
		 0.f,  1.f, 0.f
	};
	
	mappedMemory = triData;
	
	device.unmapMemory(devMemory);
	
	// associate the allocated memory with the buffer
	device.bindBufferMemory(buff, devMemory, 0);

	// make a command pool
	vk::CommandPoolCreateInfo poolInfo(vk::CommandPoolCreateFlags(), 0);
	auto commandPool = device.createCommandPool(poolInfo, vk::AllocationCallbacks::null());
	
	// make the command buffer
	vk::CommandBufferAllocateInfo commandBufferInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1); 
	auto commandBuffer = device.allocateCommandBuffers(commandBufferInfo);
	
	
	device.destroyBuffer(buff, nullptr);
	device.destroy(nullptr);
	glfwDestroyWindow(window);
	inst.destroy(nullptr);
	
	
}
