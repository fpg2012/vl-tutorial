#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdlib>

const uint32_t WIDTH = 800; // width of the window
const uint32_t HEIGHT = 600; // height of the window

// wrap all operations into this class
class HelloTriangleApplication {
public:
	HelloTriangleApplication() = default;
	void run();
private:
	void initWindow();
	void initVulkan();
	void mainLoop();
	void cleanup();

	void checkSupportedExtensions(uint32_t glfwExtensionCount, const char **glfwExtentions);

	GLFWwindow* window;
	VkInstance instance;
};

using HTA = HelloTriangleApplication;

void HTA::run() {
	initWindow(); // init window with GLFW
	initVulkan(); // init VkInstance, physical device and logical device
	mainLoop();
	cleanup();
}

// init VkInstance, physical device and logical device
void HTA::initVulkan() {
	VkApplicationInfo appInfo{
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "Hello Triangle",
		.applicationVersion = VK_MAKE_API_VERSION(1, 0, 0, 0),
		.pEngineName = "No Engine",
		.engineVersion = VK_MAKE_API_VERSION(1, 0, 0, 0),
		.apiVersion = VK_API_VERSION_1_0,
	};

	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	checkSupportedExtensions(glfwExtensionCount, glfwExtensions);
	
	VkInstanceCreateInfo createInfo{
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &appInfo,
		.enabledLayerCount = 0,
		.enabledExtensionCount = glfwExtensionCount,
		.ppEnabledExtensionNames = glfwExtensions,
	};

	VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to create instance!");
	}
}

// a bunch of vkDestroy & vkFree functions
void HTA::cleanup() {
	vkDestroyInstance(instance, nullptr);

	glfwDestroyWindow(window);
	glfwTerminate();
}

void HTA::mainLoop() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
	}
}

// init window with GLFW
void HTA::initWindow() {
	glfwInit();
	std::cout << "vulkan support: " << (glfwVulkanSupported() ? "TRUE" : "FALSE") << std::endl;
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	// create a window titled "vulkan"
	window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
}

void HTA::checkSupportedExtensions(uint32_t glfwExtensionCount, const char** glfwExtentions) {
	uint32_t extensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
	std::vector<VkExtensionProperties> extensions(extensionCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

	std::cout << "available extensions(" << extensionCount << "):\n";
	for (const auto& extension : extensions) {
		std::cout << '\t' << extension.extensionName << '\n';
	}

	std::cout << "glfw extensions(" << glfwExtensionCount << "):\n";
	for (uint32_t i = 0; i < glfwExtensionCount; ++i) {
		std::cout << '\t' << glfwExtentions[i] << '\n';
	}
}

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}