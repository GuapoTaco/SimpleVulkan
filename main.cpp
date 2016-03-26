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
	// so we can catch GLFW errors
	glfwSetErrorCallback([](int error, const char* desc)
	{
		std::cerr << "GLFW error " << error << ": " << desc << std::endl; 
	});
	
	glfwInit();
	
	if(glfwVulkanSupported() == GLFW_FALSE)
	{
		std::cerr << "This computer doesn't have vulkan support" << std::endl;
		exit(-1);
	}
	
	// we don't need a OpenGL context
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	
	// create a window through GLFW
	GLFWwindow* window = glfwCreateWindow(1280, 720, "Here", nullptr, nullptr);

	
	// create instance
	//////////////////
	vk::Instance inst;
	{
		vk::ApplicationInfo appInfo(
			"Test app", 				// application name
			0,							// application version 
			"Engine", 					// engine name
			0, 							// engine version
			VK_MAKE_VERSION(1, 0, 0)	// minimum vulkan version, use VK_MAKE_VERSION so it is properly packed into an int
		);
		
		// the extensions that the application requires
		const char* extensionNames[] = { 
			"VK_KHR_xcb_surface",	// so we can have X windows. GLFW requires this 
			"VK_KHR_surface" 		// also for GLFW, we need to acquire a surface from them
		};
		vk::InstanceCreateInfo instInfo(
			{},				// Reserved flags -- no actual need for them yet
			&appInfo,		// The application info to use
			0,				// The layer count. Use this if you are using validation layers or something
			nullptr, 		// A pointer to the layers. We aren't using layers
			2,				// The extension count
			extensionNames	// A pointer to const char*; the names of the extensions
		); 
		
		inst = vk::createInstance(
			instInfo,	// The instance info to use 
			nullptr		// The allocator--the default is fine for us
		);
	}

	// create surface
	/////////////////
	vk::SurfaceKHR surface;
	{
		VkSurfaceKHR c_surface; // the GLFW API takes in the C version, so we need to make this temporary
		auto result = glfwCreateWindowSurface(
			inst, 		// the VkInstance, our C++ type is converted into the C type
			window, 	// the GLFWWindow* to get the surface from
			nullptr, 	// the allocator--default is fine
			&c_surface	// the "return" type
		);
		
		// simple error checking
		if(result != VK_SUCCESS) throw std::runtime_error("Error creating surface with error: " + vk::to_string(vk::Result(result)));
		
		// convert the C type into the C++ type
		surface = c_surface;
	}
	
	// acquire physical device
	//////////////////////////
	vk::PhysicalDevice physical_device;
	{
		// get the physical devices from the instance
		std::vector<vk::PhysicalDevice> phyDevices;
		inst.enumeratePhysicalDevices(phyDevices);
		
		// just select the first one
		if(phyDevices.empty()) throw std::runtime_error("No suitable device found");
		
		physical_device = phyDevices[0];
	}
	
	// get the first queue family that satisifies the requirements
	//////////////////////////////////////////////////////////////
	uint32_t family_queue_index = 0;
	{
		// get the family properties
		auto queueFamilyProps = physical_device.getQueueFamilyProperties();
		for(size_t i = 0; i < queueFamilyProps.size(); ++i)
		{
			// check if it is valid for us
			if(queueFamilyProps[i].queueFlags() & vk::QueueFlagBits::eGraphics &&  	// it has to be graphics capable
				physical_device.getSurfaceSupportKHR(i, surface))					// and it has to support the surface we have
			{
				family_queue_index = i;
				break;
			}
		}
	}
	
	// create the device from physical_device
	/////////////////////////////////////////
	vk::Device device;
	{
	
	// get the queue from the device
		float queue_priorities[] = {0.0f}; // the queue should have priority 0
		
		// info for queue creation. We will create the queue in the family decided above
		vk::DeviceQueueCreateInfo queueInfo(
			{}, 				// reserved flags
			family_queue_index, // the family to create the queue in
			1, 					// the amount of queues to create--we will only make one
			queue_priorities 	// the priority of the queue
		);
		
		// the info for creating the device
		vk::DeviceCreateInfo devInfo(
			{}, 		// reserved flags
			1, 			// how many DeviceQueueCreateInfos you will pass
			&queueInfo,	// a pointer to the DeviceQueueCreateInfos you have created. You will likely want one of these for each queue family.
			0, 			// How many layers you will enable for the device. 
			nullptr, 	// A pointer to the names of the layers you want to load
			0, 			// How many extensions you will enable for the device.
			nullptr, 	// A pointer to the names of the extensions
			nullptr		// The device features to use. This is just a struct of VkBool32s, we won't enable any here.
		);
		
		// actually create the device now
		device = physical_device.createDevice(
			devInfo, 	// the DeviceCreateInfo to use
			nullptr		// the allocator--default is fine
		);
		
	}
	auto device_queue = device.getQueue(
		family_queue_index, 	// The family to get the queue from
		0						// The index of the queue inside the family to use
	);
	
	// make a command pool
	//////////////////////
	vk::CommandPool command_pool;
	{
		// The metadata we need to create a CommandPool
		vk::CommandPoolCreateInfo pool_info(
			{}, 				// reserved flags
			family_queue_index	// the queue family to use. This means we can submit command buffers in this pool to any queue in this family.
		);
		command_pool = device.createCommandPool(
			pool_info, 	// The CommandPoolCreateInfo that describes the CommandPool to create
			nullptr		// The allocator--default is find
		);
	}
	
	// make a command buffer.
	// This command buffer will be used for initalization. 
	// This is so we can have a seperate one for rendering that we can reuse each frame.
	////////////////////////
	vk::CommandBuffer init_command_buffer;
	{
		// The meatadata for allocation of the command buffer
		vk::CommandBufferAllocateInfo command_buffer_info(
			command_pool,						// The pool to create them in
			vk::CommandBufferLevel::ePrimary,	// The "level" to use TODO: explain secondary
			1									// How many command buffers to create
		); 
		
		// allocate the command buffer. This returns a `std::vector`, so just get the first element--there won't be more than that because we only created one command buffer
		init_command_buffer = device.allocateCommandBuffers(command_buffer_info)[0];
	}
	
	// All commands after this will be queued in the command buffer--not actually ran, not yet.
	init_command_buffer.begin(
		vk::CommandBufferBeginInfo( // Begin metadata
			{},		// Reserved flags
			nullptr // TODO: explain inheretance
		)
	);
	
	// create the vertex buffer
	// NOTE: this doesn't actualy allocate any memory, just creates a handle. We allocate later.
	///////////////////////////
	vk::Buffer vertex_buffer;
	{
		// the buffer creation metadata
		vk::BufferCreateInfo buffer_info(
			{}, 									// reserved flags
			sizeof(vert_buffer_data_t) * 3, 		// the size of the buffer to create. We need three verticies worth for the triange.
			vk::BufferUsageFlagBits::eVertexBuffer,	// The type. This will be used as a vertex buffer in the shader.
			vk::SharingMode::eExclusive,			// This won't be a shared resource TODO: explain more
			1, 										// The number of queue families to make sure that this is valid for TODO: check
			&family_queue_index						// The queue family
		);
		vertex_buffer = device.createBuffer(
			buffer_info, 	// The BufferCreateInfo to use
			nullptr			// The allocator--default is fine
		);
	}
	
	// create the index buffer
	// NOTE: this also doens't allocate.
	// The index buffer is for saying the indicies in the vertex buffer to draw in the triangle
	///////////////////////////////////////////////////////////////////////////////////////////
	vk::Buffer index_buffer;
	{
		// the creation metadata
		vk::BufferCreateInfo index_buffer_info(
			{}, 									// reserved flags
			sizeof(glm::uvec3), 					// the buffer flag--we only want one triangle worth
			vk::BufferUsageFlagBits::eIndexBuffer, 	// the usage--it is an index buffer
			vk::SharingMode::eExclusive, 			// no sharing TODO: explain
			1,										// the number of queue families to support TODO: factcheck
			&family_queue_index						// the pointer to the queue families
		);
		// create the buffer
		index_buffer = device.createBuffer(
			index_buffer_info, 	// the BufferCreateInfo to use
			nullptr				// the allocator--default is fine
		);
	}
	
	// make uniform buffer
	// this is for the MVP matrix we will upload to the device
	// again, this doesn't allocate
	//////////////////////////////////////////////////////////
	vk::Buffer mvp_uniform_buffer;
	{
		// buffer creation metadata
		vk::BufferCreateInfo uniBufInfo(
			{}, 										// Reserved flags
			sizeof(glm::mat4), 							// The size of the buffer
			vk::BufferUsageFlagBits::eUniformBuffer, 	// The type--this is a uniform buffer
			vk::SharingMode::eExclusive, 				// No sharing
			1,											// the number of queue families to support TODO: factcheck
			&family_queue_index							// the pointer to the queue families
		);
		// create the buffer
		mvp_uniform_buffer = device.createBuffer(
			uniBufInfo,	// the creation metadata
			nullptr		// allocator--default is fine
		);
	
	}
	
	// Allocate device memory for our three buffers. 
	// You could do two different allocation if you want, this should be less code and more efficient.
	//////////////////////////////////////////////////////////////////////////////////////////////////
	vk::DeviceMemory device_memory;
	{
		vk::MemoryAllocateInfo device_allocation_info(
				device.getBufferMemoryRequirements(vertex_buffer).size() + 		// This is the size. We use getBufferMemoryRequirements becaues this also takes into account padding, hidden metadata, etc
				device.getBufferMemoryRequirements(index_buffer).size() +		// The size is the size we need for each buffer combined
				device.getBufferMemoryRequirements(mvp_uniform_buffer).size(),
			2																	// TODO: fix & document
		); 
		// allocate
		device_memory = device.allocateMemory(
			device_allocation_info, 	// the allocation metadata
			nullptr						// allocator--default is fine
		);
	}
	
	// setup and write to all buffers
	/////////////////////////////
	{
		
		// associate the buffer to the allocated space. Otherwise, these will just be two independent objects that know nothing of each other
		device.bindBufferMemory(
			vertex_buffer, 	// the buffer object
			device_memory, 	// the allocated memory
			0				// the offset--this buffer uses the beginning of the allocated memory
		);
		device.bindBufferMemory(
			index_buffer, 												// the buffer object
			device_memory, 												// the allocated memory
			device.getBufferMemoryRequirements(vertex_buffer).size()	// the offset--vertex_buffer uses the first bit, 
																		// 		we will use the second bit for the index buffer
		);
		device.bindBufferMemory(
			mvp_uniform_buffer,				// the buffer object
			device_memory,					// the allocated memory
			device.getBufferMemoryRequirements(vertex_buffer).size() +		// offset 
				device.getBufferMemoryRequirements(index_buffer).size()
		);
		
		// write to the buffer. "map" means that every change to mapped_data updates exactly in the device.
		auto mapped_data = device.mapMemory(device_memory, 0, VK_WHOLE_SIZE, {});
		
		// the vertex data
		vert_buffer_data_t triAndColorData[] = {
			//COLOR     		 UV
			{{-1.f, -1.f, 0.f}, {1.f, 1.f}},
			{{ 1.f, -1.f, 0.f}, {0.f, 1.f}},
			{{ 0.f, 1.f,  0.f}, {.5f, 0.f}}
		};
		// copy it into the mapped data 
		memcpy(mapped_data, triAndColorData, sizeof(triAndColorData));
		
		// the indicies
		glm::uvec3 indicies = {0, 1, 2};
		
		// copy this data into the mapped data, but with the offset of the size of the vertex buffer
		memcpy((char*)mapped_data + device.getBufferMemoryRequirements(vertex_buffer).size(), &indicies, sizeof(indicies));
		
		// copy in the MVP matrix
		glm::mat4 model;
		glm::mat4 view = glm::lookAt(glm::vec3(0, 4, 2), glm::vec3(0, 0, 0), glm::vec3(0.f, 0.f, 1.f));
		glm::mat4 projection = glm::perspective(glm::radians(30.f), 1280.f/720.f, 0.f, 100.f);
		
		glm::mat4 MVP = projection * view * model;
		
		// copy the data in
		memcpy((char*)mapped_data + device.getBufferMemoryRequirements(vertex_buffer).size() + 
				device.getBufferMemoryRequirements(index_buffer).size(), 
			&MVP, sizeof(MVP));
		
		
		// stop the mapping
		device.unmapMemory(device_memory);
		
	}
	
	// create image
	// Images are quite like buffers--no allocation is done upon creation.
	// Space will be allocated later
	///////////////
	vk::Image image;
	vk::DeviceMemory image_device_memory;
	vk::ImageView image_view;
	vk::Sampler sampler;
	{
		// load the image from a file
		std::vector<unsigned char> image_data; 
		vk::Extent3D image_extents;
		{
			auto err = lodepng::decode(image_data, image_extents.width(), image_extents.height(), "image.png", LodePNGColorType::LCT_RGBA, 8);
			assert(!err); 
		}
		// this is default to 0, but we need 1 for a 2D image
		image_extents.depth(1);
		
		// create the image
		{
			// image creation metadata
			vk::ImageCreateInfo image_info(
				{}, 								// reserved flags
				vk::ImageType::e2D, 				// the image is 2D
				vk::Format::eR8G8B8A8Unorm, 		// the format for the image. This lists the components
				image_extents, 						// the size of the image--make sure the depth is 1 and not 0
				1,									// how many mip levels the image has. In a production environment you would want to have mipmaps 
				1,									// how many layers TODO: document 
				vk::SampleCountFlagBits::e1,		// how many samples the image should have TODO: document
				vk::ImageTiling::eLinear,			// how the image should tiled
				vk::ImageUsageFlagBits::eSampled,	// usage: we will be sampling it in a shader
				vk::SharingMode::eExclusive,		// no sharing
				1, 									// the count of queue families
				&family_queue_index, 				// a pointer to the queue families TODO document
				vk::ImageLayout::ePreinitialized	// TODO: not sure at all
			);
			// create the image
			image = device.createImage(
				image_info, 	// the image metadta
				nullptr			// allocator--default is fine
			);
		}
		
		// allocate & write memory for the image
		{
			// allocation metadata
			vk::MemoryAllocateInfo imageAllocInfo(
				device.getImageMemoryRequirements(image).size(), 	// allocation size
				2													// TODO: fix and document
			);
			
			
			image_device_memory = device.allocateMemory(imageAllocInfo, nullptr);
			
			// associate the allocated memory and the image
			device.bindImageMemory(image, image_device_memory, 0);
			
			// map memory
			auto mapped_data = device.mapMemory(image_device_memory, 0, VK_WHOLE_SIZE, {});
			
			// copy data in
			memcpy(mapped_data, &image_data[0], image_data.size());
			
			device.unmapMemory(image_device_memory);
		
		}
		// make an image view
		{
			vk::ImageViewCreateInfo imageViewInfo(
				{}, 									// Reserved flags
				image, 									// the image to create a view for
				vk::ImageViewType::e2D, 				// the image type
				vk::Format::eR8G8B8A8Unorm, 			// the data type
				vk::ComponentMapping(					// the component mapping--our data is already in order, so we don't need to reorder
					vk::ComponentSwizzle::eR, 	
					vk::ComponentSwizzle::eG,
					vk::ComponentSwizzle::eB,
					vk::ComponentSwizzle::eA),
				vk::ImageSubresourceRange(				// TODO: document
					vk::ImageAspectFlagBits::eColor,
					0,
					1,
					0, 
					1
				)
			);
			image_view = device.createImageView(
				imageViewInfo, 
				nullptr			// allocator--default is fine
			);
		}
		
		// create a sampler
		// This just defines how the image will be viewed TODO: factcheck
		{
			vk::SamplerCreateInfo samplerInfo(
				{}, 									// Reserved flags
				vk::Filter::eNearest, 					// the magnification filter
				vk::Filter::eNearest, 					// the minification filter
				vk::SamplerMipmapMode::eNearest,		// the mipmap mode 
				vk::SamplerAddressMode::eClampToEdge, 	// what happens when UVs are outside [0, 1]?
				vk::SamplerAddressMode::eClampToEdge,
				vk::SamplerAddressMode::eClampToEdge,
				0.f,									// TODO: miploadBias
				VK_TRUE, 								// enable antistrophy
				3.f, 									// maxiumum antistrophy
				VK_FALSE, 								// TODO: compareenable
				vk::CompareOp::eNever, 					// compareOp, doesn't matter in this case; it is disabled
				0.f,									// min LOD TODO
				0.f,									// max LOD TODO
				vk::BorderColor::eFloatOpaqueWhite, 	// border color TODO??
				VK_FALSE								// unnormalizedCordinates TODO
			);
			sampler = device.createSampler(samplerInfo, nullptr);
		}
		
	}
	
	// Create shader modules
	// These are loaded from the disk in spir-V form
	////////////////////////
	vk::ShaderModule vert_module, frag_module;
	{
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
		
		vert_module = device.createShaderModule(vertModuleInfo, nullptr);
		frag_module = device.createShaderModule(fragModuleInfo, nullptr);
	}

	
	// make descriptor set layout
	vk::DescriptorSetLayoutBinding bindings[] = {
		{0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr},
		{1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr}
	};
 	vk::DescriptorSetLayoutCreateInfo descSetLayoutCreateInfo({}, 2, bindings);
	auto descriptor_set_layout = device.createDescriptorSetLayout(descSetLayoutCreateInfo, nullptr);
	
	// make descriptor set pool
	vk::DescriptorPoolSize descPoolSizes[] = {
		{vk::DescriptorType::eUniformBuffer, 1},
		{vk::DescriptorType::eCombinedImageSampler, 1}
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
	vk::WriteDescriptorSet imageWriteDescSet(descriptor_set, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &descImageInfo, nullptr, nullptr);
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
	init_command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe, {}, {}, {}, {imageMemoryBarrier});
	
	init_command_buffer.end();
	
	// submit
	vk::SubmitInfo initSubmitInfo(0, nullptr, nullptr, 1, &init_command_buffer, 0, nullptr);
	device_queue.submit({ initSubmitInfo }, {});
	
	// wait for completion
	device_queue.waitIdle();
	
	// RENDER
	
	// make a new command buffer for the render bit 
	vk::CommandBuffer render_command_buffer;
	{
		vk::CommandBufferAllocateInfo command_buffer_info(
			command_pool,
			vk::CommandBufferLevel::ePrimary,
			1
		);
		render_command_buffer = device.allocateCommandBuffers(command_buffer_info)[0];
	}
	
	render_command_buffer.begin(vk::CommandBufferBeginInfo({}, nullptr));
	{
		// clear color on the screen
		vk::ClearValue clearColor(vk::ClearColorValue(std::array<float, 4>{{ 0.f, 0.f, 0.f, 1.f }})); 
		
		// start the render pass
		vk::RenderPassBeginInfo renderPassBeginInfo(render_pass, framebuffers[nextSwapImage], vk::Rect2D({0, 0}, { 1280, 720 }), 1, &clearColor);
		render_command_buffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
		
		render_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline);
		
		
		render_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, {descriptor_set}, {});
		
		
		// viewport
		vk::Viewport viewport(0, 0, 1280, 720, 0.f, 1.f);
		render_command_buffer.setViewport(0, {viewport});
		
		// set scissor
		vk::Rect2D scissor({0, 0}, {1280, 720});
		render_command_buffer.setScissor(0, {scissor});
		
		// bind buffer -- this binds both the location and UV
		render_command_buffer.bindVertexBuffers(0, {vertex_buffer, vertex_buffer}, {0, 0});
		
		render_command_buffer.bindIndexBuffer(index_buffer, 0, vk::IndexType::eUint32);
		
		// DRAW!!!
		render_command_buffer.drawIndexed(3, 1, 0, 0, 0);
		
		render_command_buffer.endRenderPass();
		
	}
	render_command_buffer.end();
	
	vk::Semaphore sema1 = create_semaphore(device);
	while(!glfwWindowShouldClose(window))
	{
		
		// submit render
		vk::PipelineStageFlags pipeStageFlags = vk::PipelineStageFlagBits::eBottomOfPipe;
		vk::SubmitInfo renderCommandSubmit(0, nullptr, &pipeStageFlags, 1, &render_command_buffer, 1, &sema1);
		
		
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
	device.freeCommandBuffers(command_pool, {render_command_buffer});
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
	device.freeCommandBuffers(command_pool, {init_command_buffer});
	device.destroyCommandPool(command_pool, nullptr);
	device.destroy(nullptr);
	inst.destroySurfaceKHR(surface, nullptr);
	inst.destroy(nullptr);
	glfwTerminate();
	
}
