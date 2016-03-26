#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VKCPP_ENHANCED_MODE
#include <vulkan/vk_cpp.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>
#include <thread>
#include <array>

#include "lodepng.h"

vk::Format format;

struct vert_buffer_data_t 
{
	glm::vec3 loc;
	glm::vec2 UV;
};

vk::Instance create_instance()
{
	vk::ApplicationInfo appInfo("Test app", 0, "Engine", 0, VK_MAKE_VERSION(1, 0, 0));
	
	const char* extensionNames[] = { "VK_KHR_xcb_surface", "VK_EXT_debug_report", "VK_KHR_surface" };
	const char* layerNames[] = { "VK_LAYER_LUNARG_standard_validation"};
	vk::InstanceCreateInfo instInfo({}, &appInfo, 1, layerNames, 3, extensionNames); 
	
	return vk::createInstance(instInfo, nullptr);
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
	
	auto props = phyDevices[0].getProperties();
	auto heaps = phyDevices[0].getMemoryProperties().memoryHeaps();
	auto mem_types = phyDevices[0].getMemoryProperties().memoryTypes();
	
	std::cout << "Using physical device: " << props.deviceName() << ":" << vk::to_string(props.deviceType()) << std::endl;
	std::cout << "With memory types: { ";
	for(auto i = 0; i < phyDevices[0].getMemoryProperties().memoryTypeCount(); ++i)
	{
		std::cout << "{" << mem_types[i].heapIndex() << " ," << vk::to_string(mem_types[0].propertyFlags()) << "}, ";
	}
	std::cout << " }\nWhich has heaps with sizes: { "; 
	for(auto i = 0; i < phyDevices[0].getMemoryProperties().memoryHeapCount(); ++i)
	{
		std::cout << heaps[i].size() << ", ";
	}
	std::cout << " }" << std::endl;
	
	return phyDevices[0];
	
}

vk::Device create_device(const vk::PhysicalDevice& physical_device, int family_queue_index)
{
	auto physDevQueueProps = physical_device.getQueueFamilyProperties();
	
	float queue_priorities[] = {0.0f};
	vk::DeviceQueueCreateInfo queueInfo({}, family_queue_index, physDevQueueProps[family_queue_index].queueCount(), queue_priorities);
	
	vk::DeviceCreateInfo devInfo({}, 1, &queueInfo, 0, nullptr, 0, nullptr, nullptr);
	
	return physical_device.createDevice(devInfo, nullptr);
	
}


vk::Semaphore create_semaphore(const vk::Device& device)
{
	vk::SemaphoreCreateInfo semaphoreInfo(vk::SemaphoreCreateFlags{});
	
	return device.createSemaphore(semaphoreInfo, nullptr);
}

vk::Fence create_fence(const vk::Device& device)
{
	vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlags{});
	
	return device.createFence(fenceInfo, nullptr);
}

vk::ImageView create_image_view(const vk::Device& device, const vk::Image& image)
{
	vk::ImageViewCreateInfo imageViewInfo({}, image, vk::ImageViewType::e2D, format, 
										  vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, 
															   vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA), 
										  vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 0, 0, 1));
	
	return device.createImageView(imageViewInfo, nullptr);
}

uint32_t get_correct_memory_type(vk::Device device, vk::PhysicalDevice phy_dev, vk::Buffer buff, vk::MemoryPropertyFlags flags)
{
	auto reqs = device.getBufferMemoryRequirements(buff);
	
	auto type_bits = reqs.memoryTypeBits();
	
	auto types = phy_dev.getMemoryProperties().memoryTypes();
	
	// search through the memory types
	for (uint32_t i = 0; i < phy_dev.getMemoryProperties().memoryTypeCount(); ++i)
	{
		if(types[i].propertyFlags() & flags == flags && phy_dev.getMemoryProperties().memoryHeaps()[types[i].heapIndex()].size() > reqs.size())
		{
			std::cout << "Buffer using memory type: " << i << std::endl;
			return i;
		}
	}
	
	std::cerr << "ERROR FINDING CORRECT SIZE";
	return ~0U;
	
}

uint32_t get_correct_memory_type(vk::Device device, vk::PhysicalDevice phy_dev, vk::Image image, vk::MemoryPropertyFlags flags)
{
	auto reqs = device.getImageMemoryRequirements(image);
	
	auto type_bits = reqs.memoryTypeBits();
	
	auto types = phy_dev.getMemoryProperties().memoryTypes();
	
	// search through the memory types
	for (uint32_t i = 0; i < phy_dev.getMemoryProperties().memoryTypeCount(); ++i)
	{
		if(types[i].propertyFlags() & flags == flags && phy_dev.getMemoryProperties().memoryHeaps()[types[i].heapIndex()].size() > reqs.size())
		{
			std::cout << "Image using memory type: " << i << std::endl;
			return i;
		}
	}
	
	std::cerr << "ERROR FINDING CORRECT SIZE";
	
	return ~0U;
	
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
	
	GLFWwindow* window = glfwCreateWindow(1280, 720, "Here", nullptr, nullptr);

	auto inst = create_instance();
	auto surface = create_surface(inst, window);
	auto physical_device = get_physical_device(inst);
	
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
	
	auto device = create_device(physical_device, family_queue_index);
	auto device_queue = device.getQueue(family_queue_index, 0);
	
	
	// make a command pool
	vk::CommandPoolCreateInfo poolInfo({}, family_queue_index);
	auto commandPool = device.createCommandPool(poolInfo, nullptr);
	
	// make the command buffer
	vk::CommandBufferAllocateInfo commandBufferInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1); 
	auto initCommandBuffer = device.allocateCommandBuffers(commandBufferInfo)[0];
	
	initCommandBuffer.begin(vk::CommandBufferBeginInfo({}, nullptr));
	
	// add buffer
	vk::BufferCreateInfo buffInfo({}, sizeof(vert_buffer_data_t) * 3, vk::BufferUsageFlagBits::eVertexBuffer, vk::SharingMode::eExclusive, family_queue_index, nullptr); // TODO: not sure
	auto vertex_buffer = device.createBuffer(buffInfo, nullptr);
	
	auto mem_reqs = device.getBufferMemoryRequirements(vertex_buffer);
	
	// allocate for the buffer
	vk::MemoryAllocateInfo memAllocInfo(mem_reqs.size(), get_correct_memory_type(device, physical_device, vertex_buffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)); 
	auto device_memory = device.allocateMemory(memAllocInfo, nullptr);
	
	// associate the buffer to the allocated space
	device.bindBufferMemory(vertex_buffer, device_memory, 0);
	
	// write to the buffer
	auto bufferData = device.mapMemory(device_memory, 0, sizeof(vert_buffer_data_t) * 3, {});
	
	vert_buffer_data_t triAndColorData[] = {//COLOR      UV
		{{-1.f, -1.f, 0.f}, {0.f, 0.f}},
		{{ 1.f, -1.f, 0.f}, {1.f, 0.f}},
		{{ 0.f, 1.f,  0.f}, {.5f, 1.f}}
	};
	memcpy(bufferData, triAndColorData, sizeof(triAndColorData));
	
	device.unmapMemory(device_memory);
	
	
	/// make uniform buffer
	vk::BufferCreateInfo uniBufInfo({}, sizeof(glm::mat4), vk::BufferUsageFlagBits::eUniformBuffer, vk::SharingMode::eExclusive, family_queue_index, nullptr);
	auto mvp_uniform_buffer = device.createBuffer(uniBufInfo, nullptr);
	
	// allocate
	vk::MemoryAllocateInfo uniAllocInfo(device.getBufferMemoryRequirements(mvp_uniform_buffer).size(), get_correct_memory_type(device, physical_device, mvp_uniform_buffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
	auto uniform_memory = device.allocateMemory(uniAllocInfo, nullptr);
	
	device.bindBufferMemory(mvp_uniform_buffer, uniform_memory, 0);
	
	// write
	auto uniBufferData = device.mapMemory(uniform_memory, 0, VK_WHOLE_SIZE, {});
	
	glm::mat4 model;
	glm::mat4 view = glm::lookAt(glm::vec3(3, 4, 2), glm::vec3(0, 0, 0), glm::vec3(0.f, 0.f, 1.f));
	glm::mat4 projection = glm::perspective(glm::radians(60.f), 1280.f/720.f, 0.f, 100.f);
	
	glm::mat4 MVP = projection * view * model;
	
	
	memcpy(uniBufferData, &MVP, sizeof(MVP));
	
	device.unmapMemory(uniform_memory);
	
	
	// create image
	std::vector<unsigned char> imageData; vk::Extent3D imageExtents;
	auto err = lodepng::decode(imageData, imageExtents.width(), imageExtents.height(), "image.png", LodePNGColorType::LCT_RGBA, 8);
	imageExtents.depth(1);
	assert(!err);
	
	vk::ImageCreateInfo imageInfo({}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Uint, imageExtents, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eLinear, vk::ImageUsageFlagBits::eSampled, vk::SharingMode::eExclusive, family_queue_index, nullptr, vk::ImageLayout::eGeneral);
	auto image = device.createImage(imageInfo, nullptr);
	
	std::cout << "Image Size: " << imageExtents.width() << " x " << imageExtents.height() << " x " << imageExtents.depth() << " Requested size: " << device.getImageMemoryRequirements(image).size() << " Buffer size: " << imageData.size() << std::endl;
	
	
	vk::MemoryAllocateInfo imageAllocInfo(device.getImageMemoryRequirements(image).size(), get_correct_memory_type(device, physical_device, image, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
	auto imageBuffer = device.allocateMemory(imageAllocInfo, nullptr);
	
	
	device.bindImageMemory(image, imageBuffer, 0);
	
 	auto imageBufferData = device.mapMemory(imageBuffer, 0, VK_WHOLE_SIZE, {});
// 	
// 	memcpy(imageBufferData, &imageData[0], imageData.size());
// 	
// 	device.unmapMemory(imageBuffer);
	
	// make a sampler
	vk::SamplerCreateInfo samplerInfo({}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0.f, VK_FALSE, 0.f, VK_FALSE, vk::CompareOp::eNever, 0.f, 0.f, vk::BorderColor::eFloatOpaqueWhite, VK_FALSE);
	auto sampler = device.createSampler(samplerInfo, nullptr);
	
	// make an image view
	vk::ImageViewCreateInfo imageViewInfo({}, image, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Uint, vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
	auto image_view = device.createImageView(imageViewInfo, nullptr);
	
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
	
	auto vert_module = device.createShaderModule(vertModuleInfo, nullptr);
	auto frag_module = device.createShaderModule(fragModuleInfo, nullptr);
	
	
	// make descriptor set layout
	vk::DescriptorSetLayoutBinding bindings[] = {
		{0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr},
		{1, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eVertex, nullptr}
	};
 	vk::DescriptorSetLayoutCreateInfo descSetLayoutCreateInfo({}, 2, bindings);
	auto descriptor_set_layout = device.createDescriptorSetLayout(descSetLayoutCreateInfo, nullptr);
	
	// make descriptor set pool
	vk::DescriptorPoolSize descPoolSizes[] = {
		{vk::DescriptorType::eUniformBuffer, 1}
	};
	vk::DescriptorPoolCreateInfo descPoolInfo({}, 1, 2, descPoolSizes);
	auto descriptor_pool = device.createDescriptorPool(descPoolInfo, nullptr);

	// make a descriptor set
	vk::DescriptorSetAllocateInfo descSetAllocInfo(descriptor_pool, 1, &descriptor_set_layout);
	auto descriptor_set = device.allocateDescriptorSets(descSetAllocInfo)[0];
	
	// update the descriptor set
	vk::DescriptorBufferInfo descBufferInfo(mvp_uniform_buffer, 0, sizeof(glm::mat4));
	vk::WriteDescriptorSet writeDescSet(descriptor_set, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &descBufferInfo, nullptr);
	
	vk::DescriptorImageInfo descImageInfo(sampler, image_view, vk::ImageLayout::eGeneral);
	vk::WriteDescriptorSet imageWriteDescSet(descriptor_set, 1, 0, 1, vk::DescriptorType::eSampledImage, &descImageInfo, nullptr, nullptr);
	device.updateDescriptorSets({writeDescSet, imageWriteDescSet}, {});
	
	
	
	
	
	// make pipeline layout
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &descriptor_set_layout, 0, nullptr);
	auto pipeline_layout = device.createPipelineLayout(pipelineLayoutInfo, nullptr);
	
	// make render pass
	vk::AttachmentDescription attachDesc({}, vk::Format::eR32G32B32A32Sfloat, vk::SampleCountFlagBits::e1, 
										 vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
									  vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal);
	vk::AttachmentReference colorAttachments(0, vk::ImageLayout::eColorAttachmentOptimal);
	vk::AttachmentReference depthRef(VK_ATTACHMENT_UNUSED, vk::ImageLayout::eDepthStencilAttachmentOptimal);
	vk::SubpassDescription subpassDesc({}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorAttachments, nullptr, &depthRef, 0, nullptr);
	vk::RenderPassCreateInfo renderPassInfo({}, 1, &attachDesc, 1, &subpassDesc, 0, nullptr);
	
	auto render_pass = device.createRenderPass(renderPassInfo, nullptr);
	
	
	// construct a fence for sync
	vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlags{});
	auto fence = device.createFence(fenceInfo, nullptr);
	
	
	// create the swapchain	
	std::vector<vk::SurfaceFormatKHR> formats;
	physical_device.getSurfaceFormatsKHR(surface, formats);
	
	format = formats[0].format();
	
	vk::SwapchainCreateInfoKHR swapchainInfo({}, surface, 2, format,
											 formats[0].colorSpace(), vk::Extent2D(1280, 720), 1, 
											 vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive, 0, nullptr, 
											 vk::SurfaceTransformFlagBitsKHR::eIdentity, vk::CompositeAlphaFlagBitsKHR::eOpaque, 
											 vk::PresentModeKHR::eMailboxKHR, VK_FALSE, {});

	
	auto swapchain = device.createSwapchainKHR(swapchainInfo, nullptr);
	
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
		vk::FramebufferCreateInfo framebufferInfo({}, render_pass, 1, &image_views[image_views.size() - 1], 1280, 720, 1);
		framebuffers.push_back(device.createFramebuffer(framebufferInfo, nullptr));
		
	}
	
	// get next swap image
	uint32_t nextSwapImage;
	device.acquireNextImageKHR(swapchain, UINT64_MAX, {}, {}, nextSwapImage);
	
	
	// make graphics pipeline: HOLY SHIT THAT'S A LOT OF METADATA
	vk::PipelineShaderStageCreateInfo pipeShaderStageInfo[] = {
		{{}, vk::ShaderStageFlagBits::eVertex, vert_module, "main", nullptr},
		{{}, vk::ShaderStageFlagBits::eFragment, frag_module, "main", nullptr}
	}; // GOOD GOOD
	vk::VertexInputAttributeDescription vertInputAttrDescs[] = {
		{0, 0, vk::Format::eR32G32B32Sfloat, 0}, // location
		{1, 1, vk::Format::eR32G32Sfloat, sizeof(glm::vec3)} // UVs
	}; 
	vk::VertexInputBindingDescription vertInputBindingDescs[] = {
		{0, sizeof(vert_buffer_data_t), vk::VertexInputRate::eVertex},
		{1, sizeof(vert_buffer_data_t), vk::VertexInputRate::eVertex} 
	};
	vk::PipelineVertexInputStateCreateInfo pipeVertexInputStateInfo({}, 2, vertInputBindingDescs, 2, vertInputAttrDescs); // GOOD GOOD
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
	auto graphics_pipeline = device.createGraphicsPipelines({}, {graphicsPipeInfo}, nullptr)[0];
	
	
	// make image memory barrier
	vk::ImageMemoryBarrier imageMemoryBarrier({}, vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR, family_queue_index, family_queue_index, images[nextSwapImage], vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
	initCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe, {}, {}, {}, {imageMemoryBarrier});
	
	initCommandBuffer.end();
	
	// submit
	vk::SubmitInfo initSubmitInfo(0, nullptr, nullptr, 1, &initCommandBuffer, 0, nullptr);
	device_queue.submit({ initSubmitInfo }, {});
	
	// wait for completion
	device_queue.waitIdle();
	
	// RENDER
	
	// make a new command buffer for the render bit 
	auto renderCommandBuffer = device.allocateCommandBuffers(commandBufferInfo)[0];
	
	renderCommandBuffer.begin(vk::CommandBufferBeginInfo({}, nullptr));
	{
		// clear color on the screen
		vk::ClearValue clearColor(vk::ClearColorValue(std::array<float, 4>{{ 1.f, 1.f, 0.f, 1.f }})); 
		
		// start the render pass
		vk::RenderPassBeginInfo renderPassBeginInfo(render_pass, framebuffers[nextSwapImage], vk::Rect2D({0, 0}, { 1280, 720 }), 1, &clearColor);
		renderCommandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
		
		renderCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline);
		
		
		renderCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, {descriptor_set}, {});
		
		
		// viewport
		vk::Viewport viewport(0, 0, 1280, 720, 0.f, 1.f);
		renderCommandBuffer.setViewport(0, {viewport});
		
		// set scissor
		vk::Rect2D scissor({0, 0}, {1280, 720});
		renderCommandBuffer.setScissor(0, {scissor});
		
		// bind buffer -- this binds both the location and UV
		renderCommandBuffer.bindVertexBuffers(0, {vertex_buffer, vertex_buffer}, {0, 0});
		
		// DRAW!!!
		renderCommandBuffer.draw(3, 1, 0, 0);
	
		
		renderCommandBuffer.endRenderPass();
		
	}
	renderCommandBuffer.end();
	
	vk::Semaphore sema1 = create_semaphore(device);
	while(!glfwWindowShouldClose(window))
	{
		
		// submit render
		vk::PipelineStageFlags pipeStageFlags = vk::PipelineStageFlagBits::eBottomOfPipe;
		vk::SubmitInfo renderCommandSubmit(0, nullptr, &pipeStageFlags, 1, &renderCommandBuffer, 1, &sema1);
		
		
		device_queue.submit({renderCommandSubmit}, fence);
		
		device.waitForFences({fence}, VK_TRUE, UINT64_MAX);
		
		// swap buffers
		vk::Result res;
		vk::PresentInfoKHR presentInfo(1, &sema1, 1, &swapchain, &nextSwapImage, &res);
		device_queue.presentKHR(presentInfo);
		
		if((glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL)) && glfwGetKey(window, GLFW_KEY_Q)) 
		{
			break;
		}
		
		glfwPollEvents();
		
		
		
	}
	
	// do this asap so the program still seems respsnsive
	glfwDestroyWindow(window);
	
	// cleanup
	device.destroySemaphore(sema1, nullptr);
	device.freeCommandBuffers(commandPool, {renderCommandBuffer});
	device.destroyPipeline(graphics_pipeline, nullptr);
	for(auto&& image_view : image_views)
	{
		device.destroyImageView(image_view, nullptr);
	}
	for(auto&& framebuffer : framebuffers)
	{
		device.destroyFramebuffer(framebuffer, nullptr);
	}
	device.freeDescriptorSets(descriptor_pool, {descriptor_set});
	device.destroyDescriptorPool(descriptor_pool, nullptr);
	device.destroySwapchainKHR(swapchain, nullptr);
	device.destroyFence(fence, nullptr);
	device.destroyRenderPass(render_pass, nullptr);
	device.destroyPipelineLayout(pipeline_layout, nullptr);
	device.destroyDescriptorSetLayout(descriptor_set_layout, nullptr);
	device.destroyShaderModule(frag_module, nullptr);
	device.destroyShaderModule(vert_module, nullptr);
	device.freeMemory(device_memory, nullptr);
	device.destroyBuffer(vertex_buffer, nullptr);
	device.freeCommandBuffers(commandPool, {initCommandBuffer});
	device.destroyCommandPool(commandPool, nullptr);
	device.destroy(nullptr);
	inst.destroySurfaceKHR(surface, nullptr);
	inst.destroy(nullptr);
	glfwTerminate();
	
}
