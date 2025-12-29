### README
This program demonstrates a Vulkan issue / NVIDIA driver issue where repeated calls to `vkResetCommandPool` lead to increased memory usage and slower performance over time. The commands in the pools seem to be never recycled properly?

The issue has been observed on NVIDIA GPUs and is discussed in the NVIDIA developer forums.

https://forums.developer.nvidia.com/t/vkresetcommandpool-seems-to-slow-down-overtime-and-eats-up-memory/355921