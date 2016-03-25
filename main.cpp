#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VKCPP_ENHANCED_MODE
#include <vulkan/vk_cpp.h>

#include <iostream>
#include <fstream>
#include <thread>

vk::Format format;

vk::Instance create_instance()
{
	vk::ApplicationInfo appInfo("Test app", 0, "Engine", 0, VK_MAKE_VERSION(1, 0, 0));
	
	const char* extensionNames[] = { "VK_KHR_xcb_surface", "VK_EXT_debug_report", "VK_KHR_surface" };
	vk::InstanceCreateInfo instInfo({}, &appInfo, 0, nullptr, 3, extensionNames); 
	
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

vk::Device create_device(const vk::PhysicalDevice& physical_device, int family_queue_index)
{
	auto physDevQueueProps = physical_device.getQueueFamilyProperties();
	
	float queue_priorities[] = {0.0f};
	vk::DeviceQueueCreateInfo queueInfo({}, family_queue_index, physDevQueueProps[family_queue_index].queueCount(), queue_priorities);
	
	vk::DeviceCreateInfo devInfo({}, 1, &queueInfo, 0, nullptr, 0, nullptr, nullptr);
	
	return physical_device.createDevice(devInfo, vk::AllocationCallbacks::null());
	
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
	vk::ImageViewCreateInfo imageViewInfo({}, image, vk::ImageViewType::e2D, format, 
										  vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, 
															   vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA), 
										  vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 0, 0, 1));
	
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
	
	int family_queue_index = 0;
	auto queueFamilyProps = physical_device.getQueueFamilyProperties();
	for(size_t i = 0; i < queueFamilyProps.size(); ++i)
	{
		if(queueFamilyProps[i].queueFlags() & vk::QueueFlagBits::eGraphics && physical_device.getSurfaceSupportKHR(i, surface))
		{
			family_queue_index = i;
		}
	}
	std::cout << "using family queue index " << family_queue_index << std::endl;
	
	auto device = create_device(physical_device, family_queue_index); assert(device);
	auto device_queue = device.getQueue(family_queue_index, 0); assert(device_queue);
	
	
	// make a command pool
	vk::CommandPoolCreateInfo poolInfo({}, family_queue_index);
	auto commandPool = device.createCommandPool(poolInfo, vk::AllocationCallbacks::null());
	
	// make the command buffer
	vk::CommandBufferAllocateInfo commandBufferInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1); 
	auto initCommandBuffer = device.allocateCommandBuffers(commandBufferInfo)[0];
	
	initCommandBuffer.begin(vk::CommandBufferBeginInfo({}, nullptr));
	
	// add buffer
	vk::BufferCreateInfo buffInfo({}, sizeof(float) * 9, vk::BufferUsageFlagBits::eVertexBuffer, vk::SharingMode::eExclusive, 0, nullptr); // TODO: not sure
	auto vertex_buffer = device.createBuffer(buffInfo, vk::AllocationCallbacks::null());
	
	auto mem_reqs = device.getBufferMemoryRequirements(vertex_buffer);
	
	// allocate for the buffer
	vk::MemoryAllocateInfo memAllocInfo(mem_reqs.size(), 9); // TODO: better selection of memory type
	auto device_memory = device.allocateMemory(memAllocInfo, vk::AllocationCallbacks::null());
	
	// associate the buffer to the allocated space
	device.bindBufferMemory(vertex_buffer, device_memory, 0);
	
	// write to the buffer
	auto bufferData = device.mapMemory(device_memory, 0, mem_reqs.size(), {});
	
	float triData[] = {
		-1.f, -1.f, 0.f,
		 1.f, -1.f, 0.f,
		 0.f,  1.f, 0.f
	};
	memcpy(bufferData, triData, sizeof(triData));
	
	device.unmapMemory(device_memory);
	
	
	
	// upload shaders
	std::ifstream fragFile("frag.spv", std::ios::binary | std::ios::ate);
	std::streamsize fragSize = fragFile.tellg();
	fragFile.seekg(0, std::ios::beg);
	std::vector<char> fragSpirVData(fragSize);
	fragFile.read(fragSpirVData.data(), fragSize);
	
	
	std::ifstream vertFile("vert.spv", std::ios::binary | std::ios::ate);
	std::streamsize vertSize = vertFile.tellg();
	vertFile.seekg(0, std::ios::beg);
	std::vector<char> vertSpirVData(vertSize);
	vertFile.read(vertSpirVData.data(), vertSize);
	
	vk::ShaderModuleCreateInfo vertModuleInfo({}, vertSpirVData.size(), reinterpret_cast<const uint32_t*>(vertSpirVData.data()));
	vk::ShaderModuleCreateInfo fragModuleInfo({}, fragSpirVData.size(), reinterpret_cast<const uint32_t*>(fragSpirVData.data()));
	
	auto vert_module = device.createShaderModule(vertModuleInfo, vk::AllocationCallbacks::null()); assert(vert_module);
	auto frag_module = device.createShaderModule(fragModuleInfo, vk::AllocationCallbacks::null()); assert(frag_module);
	
	
	// make descriptor set
	vk::DescriptorSetLayoutCreateInfo descSetLayoutCreateInfo({}, 0, nullptr);
	auto descriptor_set_layout = device.createDescriptorSetLayout(descSetLayoutCreateInfo, vk::AllocationCallbacks::null()); assert(descriptor_set_layout);

	// make pipeline layout
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &descriptor_set_layout, 0, nullptr);
	auto pipeline_layout = device.createPipelineLayout(pipelineLayoutInfo, vk::AllocationCallbacks::null()); assert(pipeline_layout);
	
	// make render pass
	vk::AttachmentDescription attachDesc({}, vk::Format::eR32G32B32A32Sfloat, vk::SampleCountFlagBits::e1, 
										 vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
									  vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal);
	vk::AttachmentReference colorAttachments(0, vk::ImageLayout::eColorAttachmentOptimal);
	vk::AttachmentReference depthRef(VK_ATTACHMENT_UNUSED, vk::ImageLayout::eDepthStencilAttachmentOptimal);
	vk::SubpassDescription subpassDesc({}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorAttachments, nullptr, &depthRef, 0, nullptr);
	vk::RenderPassCreateInfo renderPassInfo({}, 1, &attachDesc, 1, &subpassDesc, 0, nullptr);
	
	auto render_pass = device.createRenderPass(renderPassInfo, vk::AllocationCallbacks::null()); assert(render_pass);
	
	
	// construct a fence for sync
	vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlags{});
	auto fence = device.createFence(fenceInfo, vk::AllocationCallbacks::null());
	
	
	// create the swapchain	
	std::vector<vk::SurfaceFormatKHR> formats;
	physical_device.getSurfaceFormatsKHR(surface, formats);
	
	format = formats[0].format();
	
	vk::SwapchainCreateInfoKHR swapchainInfo({}, surface, 2, format,
											 formats[0].colorSpace(), vk::Extent2D(1280, 720), 1, 
											 vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive, 0, nullptr, 
											 vk::SurfaceTransformFlagBitsKHR::eIdentity, vk::CompositeAlphaFlagBitsKHR::eOpaque, 
											 vk::PresentModeKHR::eMailboxKHR, VK_FALSE, {});

	
	auto swapchain = device.createSwapchainKHR(swapchainInfo, vk::AllocationCallbacks::null());
	
	// get images
	std::vector<vk::Image> images;
	device.getSwapchainImagesKHR(swapchain, images);
	
	std::cout << images.size() << " Images avaliable" << std::endl;
	
	
	// make framebuffers
	std::vector<vk::Framebuffer> framebuffers;
	std::vector<vk::ImageView> image_views;
	framebuffers.reserve(images.size());
	
	for(auto&& img : images)
	{
		image_views.push_back(create_image_view(device, img));
		
		
		// make framebuffer
		vk::FramebufferCreateInfo framebufferInfo({}, render_pass, 1, &image_views[image_views.size() - 1], 1024, 720, 1);
		framebuffers.push_back(device.createFramebuffer(framebufferInfo, vk::AllocationCallbacks::null()));
		
	}
	
	// get next swap image
	uint32_t nextSwapImage;
	device.acquireNextImageKHR(swapchain, UINT64_MAX, {}, {}, nextSwapImage);
	
	
	// make graphics pipeline: HOLY SHIT THAT'S A LOT OF METADATA
	vk::PipelineShaderStageCreateInfo pipeShaderStageInfo[] = {
		{{}, vk::ShaderStageFlagBits::eVertex, vert_module, "main", nullptr},
		{{}, vk::ShaderStageFlagBits::eFragment, frag_module, "main", nullptr}
	}; // GOOD GOOD
	vk::VertexInputAttributeDescription vertInputAttrDesc(0, 0, vk::Format::eR32G32B32Sfloat, 0); // GOOD GOOD
	vk::VertexInputBindingDescription vertInputBindingDesc(0, sizeof(float) * 3, vk::VertexInputRate::eVertex); // GOOD GOOD
	vk::PipelineVertexInputStateCreateInfo pipeVertexInputStateInfo({}, 1, &vertInputBindingDesc, 1, &vertInputAttrDesc); // GOOD GOOD
	vk::PipelineInputAssemblyStateCreateInfo pipeInputAsmStateInfo({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE); // GOOD GOOD
	vk::PipelineViewportStateCreateInfo pipeViewportStateInfo({}, 0, nullptr, 1, nullptr); // GOOD GOOD
	vk::PipelineRasterizationStateCreateInfo pipeRasterizationStateInfo({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise, VK_FALSE, 0.f, 0.f, 0.f, 0.f); // GOOD GOOD
	vk::PipelineMultisampleStateCreateInfo pipeMultisampleStateInfo({}, vk::SampleCountFlagBits::e1, VK_FALSE, 0.f, nullptr, VK_FALSE, VK_FALSE); // GOOD GOOD
	vk::PipelineColorBlendAttachmentState pipeColorAttachState(VK_FALSE, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
	vk::PipelineColorBlendStateCreateInfo pipeColorBlendStateInfo({}, VK_FALSE, vk::LogicOp::eClear, 1, &pipeColorAttachState, {{0, 0, 0, 0}});
	vk::DynamicState dynStates[] = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor
	};
	vk::PipelineDynamicStateCreateInfo dynStateInfo({}, 2, dynStates);
	vk::GraphicsPipelineCreateInfo graphicsPipeInfo({}, 2, pipeShaderStageInfo, &pipeVertexInputStateInfo, &pipeInputAsmStateInfo, nullptr, &pipeViewportStateInfo, &pipeRasterizationStateInfo, &pipeMultisampleStateInfo, nullptr, &pipeColorBlendStateInfo, &dynStateInfo, pipeline_layout, render_pass, 0, {}, -1); 
	auto graphics_pipeline = device.createGraphicsPipelines({}, {graphicsPipeInfo}, vk::AllocationCallbacks::null())[0]; assert(graphics_pipeline);
	
	
	// make image memory barrier
	vk::ImageMemoryBarrier imageMemoryBarrier({}, vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR, family_queue_index, family_queue_index, images[nextSwapImage], vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
	initCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe, {}, {}, {}, {imageMemoryBarrier});
	
	initCommandBuffer.end();
	
	// submit
	vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &initCommandBuffer, 0, nullptr);
	device_queue.submit({ submitInfo }, {});
	
	// wait for completion
	device_queue.waitIdle();
	
	// RENDER
	
	// make a new command buffer for the render bit 
	auto renderCommandBuffer = device.allocateCommandBuffers(commandBufferInfo)[0];
	
	renderCommandBuffer.begin(vk::CommandBufferBeginInfo({}, nullptr));
	{
		// clear color on the screen
		vk::ClearValue clearColor(vk::ClearColorValue(std::array<float, 4>{{ 1.f, 0.f, 0.f, 1.f }})); // GOOD GOOD
		
		// start the render pass
		vk::RenderPassBeginInfo renderPassBeginInfo(render_pass, framebuffers[nextSwapImage], vk::Rect2D({0, 0}, { 1024, 720 }), 1, &clearColor); // GOOD GOOD
		renderCommandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline); // GOOD GOOD
		
		renderCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline);
		
		// viewport
		vk::Viewport viewport(0, 0, 1280, 720, 0.f, 1.f);
		renderCommandBuffer.setViewport(0, {viewport});
		
		// set scissor
		vk::Rect2D scissor({0, 0}, {1280, 720});
		renderCommandBuffer.setScissor(0, {scissor});
		
		// bind buffer
		renderCommandBuffer.bindVertexBuffers(0, {vertex_buffer}, { 0 });
		
		// DRAW!!!
		renderCommandBuffer.draw(3, 1, 0, 0);
	}
	renderCommandBuffer.end();
	
	// submit render
	vk::PipelineStageFlags pipeStageFlags = vk::PipelineStageFlagBits::eBottomOfPipe;
	vk::SubmitInfo renderCommandSubmit(0, nullptr, &pipeStageFlags, 1, &renderCommandBuffer, 0, nullptr);
	device_queue.submit({submitInfo}, {});
	
	// swap buffers
	vk::Result res;
	vk::PresentInfoKHR presentInfo(0, nullptr, 1, &swapchain, &nextSwapImage, &res);
	device_queue.presentKHR(presentInfo);
	
	std::this_thread::sleep_for(std::chrono::seconds(10));
	
	
}
