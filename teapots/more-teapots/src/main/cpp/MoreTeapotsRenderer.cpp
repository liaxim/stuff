/*
 * Copyright 2013 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//--------------------------------------------------------------------------------
// MoreTeapotsRenderer.cpp
// Render teapots
//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
// Include files
//--------------------------------------------------------------------------------
#include "MoreTeapotsRenderer.h"

#include <string.h>
#include <android/hardware_buffer.h>
#include <EGL/eglext.h>
#include <GLES2/gl2ext.h>
#include <vulkan/vulkan.h>

//--------------------------------------------------------------------------------
// Teapot model data
//--------------------------------------------------------------------------------
#include "teapot.inl"

constexpr int width = 1000;
constexpr int height = 1000;

//--------------------------------------------------------------------------------
// Ctor
//--------------------------------------------------------------------------------
MoreTeapotsRenderer::MoreTeapotsRenderer()
    : geometry_instancing_support_(false) {}

//--------------------------------------------------------------------------------
// Dtor
//--------------------------------------------------------------------------------
MoreTeapotsRenderer::~MoreTeapotsRenderer() { Unload(); }

bool getMemoryTypeIndex(const VkPhysicalDeviceMemoryProperties &memoryProperties,
                                               const VkMemoryRequirements &memoryRequirements, uint32_t &memoryTypeIndex) {
    bool foundMemoryType = false;

    VkMemoryPropertyFlags requiredProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const uint32_t memoryCount = memoryProperties.memoryTypeCount;
    for (uint32_t memoryIndex = 0; memoryIndex < memoryCount; ++memoryIndex) {
        const uint32_t memoryTypeBits = (1 << memoryIndex);
        const bool isRequiredMemoryType = memoryRequirements.memoryTypeBits & memoryTypeBits;

        const VkMemoryPropertyFlags properties =
                memoryProperties.memoryTypes[memoryIndex].propertyFlags;
        const bool hasRequiredProperties =
                (properties & requiredProperties) == requiredProperties;

        if (isRequiredMemoryType && hasRequiredProperties) {
            foundMemoryType = true;
            memoryTypeIndex = memoryIndex;
        }
    }

    return foundMemoryType;
}

int getMemoryFileDescriptor(VkInstance instance, VkDevice device, VkDeviceMemory deviceMemory) {
  int fileDescriptor;

  auto mVkGetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetInstanceProcAddr(
          instance, "vkGetMemoryFdKHR"));

  const VkMemoryGetFdInfoKHR memoryGetFdInfo{
          .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
          .memory = deviceMemory,
          .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
  VkResult result = mVkGetMemoryFdKHR(device, &memoryGetFdInfo, &fileDescriptor);
  assert(result == VK_SUCCESS);

  return fileDescriptor;
}



//--------------------------------------------------------------------------------
// Init
//--------------------------------------------------------------------------------
void MoreTeapotsRenderer::Init(const int32_t numX, const int32_t numY,
                               const int32_t numZ) {
  if (ndk_helper::GLContext::GetInstance()->GetGLVersion() >= 3.0) {
    geometry_instancing_support_ = true;
  } else if (ndk_helper::GLContext::GetInstance()->CheckExtension(
                 "GL_NV_draw_instanced") &&
             ndk_helper::GLContext::GetInstance()->CheckExtension(
                 "GL_NV_uniform_buffer_object")) {
    LOGI("Supported via extension!");
    //_bGeometryInstancingSupport = true;
    //_bARBSupport = true; //Need to patch shaders
    // Currently this has been disabled
  }

  // Settings
  glFrontFace(GL_CCW);

  // Create Index buffer
  num_indices_ = sizeof(teapotIndices) / sizeof(teapotIndices[0]);
  glGenBuffers(1, &ibo_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(teapotIndices), teapotIndices,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  // Create VBO
  num_vertices_ = sizeof(teapotPositions) / sizeof(teapotPositions[0]) / 3;
  int32_t stride = sizeof(TEAPOT_VERTEX);
  int32_t index = 0;
  TEAPOT_VERTEX* p = new TEAPOT_VERTEX[num_vertices_];
  for (int32_t i = 0; i < num_vertices_; ++i) {
    p[i].pos[0] = teapotPositions[index];
    p[i].pos[1] = teapotPositions[index + 1];
    p[i].pos[2] = teapotPositions[index + 2];

    p[i].normal[0] = teapotNormals[index];
    p[i].normal[1] = teapotNormals[index + 1];
    p[i].normal[2] = teapotNormals[index + 2];
    index += 3;
  }
  glGenBuffers(1, &vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, stride * num_vertices_, p, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  delete[] p;

  // Init Projection matrices
  teapot_x_ = numX;
  teapot_y_ = numY;
  teapot_z_ = numZ;
  vec_mat_models_.reserve(teapot_x_ * teapot_y_ * teapot_z_);

  UpdateViewport();

  const float total_width = 500.f;
  float gap_x = total_width / (teapot_x_ - 1);
  float gap_y = total_width / (teapot_y_ - 1);
  float gap_z = total_width / (teapot_z_ - 1);
  float offset_x = -total_width / 2.f;
  float offset_y = -total_width / 2.f;
  float offset_z = -total_width / 2.f;

  for (int32_t x = 0; x < teapot_x_; ++x)
    for (int32_t y = 0; y < teapot_y_; ++y)
      for (int32_t z = 0; z < teapot_z_; ++z) {
        vec_mat_models_.push_back(ndk_helper::Mat4::Translation(
            x * gap_x + offset_x, y * gap_y + offset_y,
            z * gap_z + offset_z));
        vec_colors_.push_back(ndk_helper::Vec3(
            random() / float(RAND_MAX * 1.1), random() / float(RAND_MAX * 1.1),
            random() / float(RAND_MAX * 1.1)));

        float rotation_x = random() / float(RAND_MAX) - 0.5f;
        float rotation_y = random() / float(RAND_MAX) - 0.5f;
        vec_rotations_.push_back(ndk_helper::Vec2(rotation_x * 0.05f, rotation_y * 0.05f));
        vec_current_rotations_.push_back(
            ndk_helper::Vec2(rotation_x * M_PI, rotation_y * M_PI));
      }

  if (geometry_instancing_support_) {
    //
    // Create parameter dictionary for shader patch
    std::map<std::string, std::string> param;
    param[std::string("%NUM_TEAPOT%")] =
        ToString(teapot_x_ * teapot_y_ * teapot_z_);
    param[std::string("%LOCATION_VERTEX%")] = ToString(ATTRIB_VERTEX);
    param[std::string("%LOCATION_NORMAL%")] = ToString(ATTRIB_NORMAL);
    if (arb_support_)
      param[std::string("%ARB%")] = std::string("ARB");
    else
      param[std::string("%ARB%")] = std::string("");

    // Load shader
    bool b = LoadShadersES3(&shader_param_, "Shaders/VS_ShaderPlainES3.vsh",
                            "Shaders/ShaderPlainES3.fsh", param);
    if (b) {
      //
      // Create uniform buffer
      //
      GLuint bindingPoint = 1;
      GLuint blockIndex;
      blockIndex = glGetUniformBlockIndex(shader_param_.program_, "ParamBlock");
      glUniformBlockBinding(shader_param_.program_, blockIndex, bindingPoint);

      // Retrieve array stride value
      int32_t num_indices;
      glGetActiveUniformBlockiv(shader_param_.program_, blockIndex,
                                GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &num_indices);
      GLint i[num_indices];
      GLint stride[num_indices];
      glGetActiveUniformBlockiv(shader_param_.program_, blockIndex,
                                GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, i);
      glGetActiveUniformsiv(shader_param_.program_, num_indices, (GLuint*)i,
                            GL_UNIFORM_ARRAY_STRIDE, stride);

      ubo_matrix_stride_ = stride[0] / sizeof(float);
      ubo_vector_stride_ = stride[2] / sizeof(float);

      glGenBuffers(1, &ubo_);
      glBindBuffer(GL_UNIFORM_BUFFER, ubo_);
      glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, ubo_);

      // Store color value which wouldn't be updated every frame
      int32_t size = teapot_x_ * teapot_y_ * teapot_z_ *
                      (ubo_matrix_stride_ + ubo_matrix_stride_ +
                       ubo_vector_stride_);  // Mat4 + Mat4 + Vec3 + 1 stride
      float* pBuffer = new float[size];
      float* pColor =
          pBuffer + teapot_x_ * teapot_y_ * teapot_z_ * ubo_matrix_stride_ * 2;
      for (int32_t i = 0; i < teapot_x_ * teapot_y_ * teapot_z_; ++i) {
        memcpy(pColor, &vec_colors_[i], 3 * sizeof(float));
        pColor += ubo_vector_stride_;  // Assuming std140 layout which is 4
                                       // DWORD stride for vectors
      }

      glBufferData(GL_UNIFORM_BUFFER, size * sizeof(float), pBuffer,
                   GL_DYNAMIC_DRAW);
      delete[] pBuffer;
    } else {
      LOGI("Shader compilation failed!! Falls back to ES2.0 pass");
      // This happens some devices.
      geometry_instancing_support_ = false;
      // Load shader for GLES2.0
      LoadShaders(&shader_param_, "Shaders/VS_ShaderPlain.vsh",
                  "Shaders/ShaderPlain.fsh");
    }
  } else {
    // Load shader for GLES2.0
    LoadShaders(&shader_param_, "Shaders/VS_ShaderPlain.vsh",
                "Shaders/ShaderPlain.fsh");
  }

  ///create
    VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    VkInstance vulkanInstance;
    auto result = vkCreateInstance(&instInfo, nullptr, &vulkanInstance);
    uint32_t count;
    result = vkEnumeratePhysicalDevices(vulkanInstance, &count, nullptr);
    VkPhysicalDevice physicalDevices[count];
    result = vkEnumeratePhysicalDevices(vulkanInstance, &count, physicalDevices);

    auto physicalDevice = physicalDevices[0];
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    //queuefamilyindex
    uint32_t queueFamilyIndex;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProps(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProps.data());
    bool foundQueueFamilyIndex = false;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if ((queueFamilyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0u) {
            queueFamilyIndex = i;
            foundQueueFamilyIndex = true;
            break;
        }
    }

    //createdevice
    float queuePriorities = 0;
    VkDeviceQueueCreateInfo queueInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriorities
    };
    VkDeviceCreateInfo deviceInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueInfo,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = nullptr
    };
    VkDevice device;
    result = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);
    assert(result == VK_SUCCESS);

    //createImage
    const VkExternalMemoryImageCreateInfo externalMemoryImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };

    //!TODO: Map remaining XR flags.
    const VkImageCreateInfo imageInfo {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &externalMemoryImageCreateInfo,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent.width = 1024,
            .extent.height = 1024,
            .extent.depth = 1,
            .mipLevels = 1,
            .arrayLayers = 2,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            //.tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,
            //.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            //.queueFamilyIndexCount = 0,
            //.pQueueFamilyIndices = nullptr,
            //.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VkImage vkimage;
    result = vkCreateImage(device, &imageInfo, nullptr, &vkimage);
    assert(result == VK_SUCCESS);

    //allocatememory
    VkMemoryRequirements memoryRequirements{};
    vkGetImageMemoryRequirements(device, vkimage, &memoryRequirements);

    uint32_t memoryTypeIndex;
    bool foundMemoryType = getMemoryTypeIndex(memoryProperties, memoryRequirements,
                                              memoryTypeIndex);
    assert(foundMemoryType);

    VkExportMemoryAllocateInfo exportMemoryAllocateInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportMemoryAllocateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryDedicatedAllocateInfo dedicatedAllocInfo{
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedAllocInfo.pNext = &exportMemoryAllocateInfo;
    dedicatedAllocInfo.image = vkimage;
    dedicatedAllocInfo.buffer = VK_NULL_HANDLE;

    const VkMemoryAllocateInfo memoryAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &dedicatedAllocInfo,
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = memoryTypeIndex
    };

    VkDeviceMemory deviceMemory;
    result = vkAllocateMemory(device, &memoryAllocateInfo, NULL, &deviceMemory);
    assert(result == VK_SUCCESS);
    auto memorySize = memoryRequirements.size;
    result = vkBindImageMemory(device, vkimage, deviceMemory, 0);
    assert(result == VK_SUCCESS);
    auto fileDescriptor = getMemoryFileDescriptor(vulkanInstance, device, deviceMemory);

    GLuint memoryObject;
    static const auto pGlCreateMemoryObjectsEXT = reinterpret_cast<PFNGLCREATEMEMORYOBJECTSEXTPROC>(eglGetProcAddress(
             "glCreateMemoryObjectsEXT"));
    pGlCreateMemoryObjectsEXT(1, &memoryObject);
    auto glerror = glGetError();

    GLint dedicatedMemory = GL_TRUE;
    static const auto pGlMemoryObjectParameterivEXT = reinterpret_cast<PFNGLMEMORYOBJECTPARAMETERIVEXTPROC>(eglGetProcAddress(
              "glMemoryObjectParameterivEXT"));
    pGlMemoryObjectParameterivEXT(memoryObject, GL_DEDICATED_MEMORY_OBJECT_EXT,
                                    &dedicatedMemory);
    glerror = glGetError();

    static const auto pGlImportMemoryFdEXT = reinterpret_cast<PFNGLIMPORTMEMORYFDEXTPROC>(eglGetProcAddress(
              "glImportMemoryFdEXT"));
      // A successful import operation transfers ownership of fileDescriptor to the GL implementation
    pGlImportMemoryFdEXT(memoryObject, memorySize, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fileDescriptor);
    glerror = glGetError();


    glGenTextures(1, &mColorTexture);
    GLenum textureTarget = GL_TEXTURE_2D_ARRAY;
    glBindTexture(textureTarget, mColorTexture);

    static const auto pGlTexStorageMem3DEXT =
            reinterpret_cast<PFNGLTEXSTORAGEMEM3DEXTPROC>(eglGetProcAddress("glTexStorageMem3DEXT"));
    pGlTexStorageMem3DEXT(
            GL_TEXTURE_2D_ARRAY,
            1,
            GL_RGBA8,
            1024,
            1024,
            2,
            memoryObject,
            0);
    glerror = glGetError();

    glTexParameteri(textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(textureTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(textureTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(textureTarget, 0);

  glGenFramebuffers(1, &mFramebuffer);

  vkDestroyImage(device, vkimage, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(vulkanInstance, nullptr);
}

void MoreTeapotsRenderer::UpdateViewport() {
  int32_t viewport[4];
  glGetIntegerv(GL_VIEWPORT, viewport);

  const float CAM_NEAR = 5.f;
  const float CAM_FAR = 10000.f;
  if (viewport[2] < viewport[3]) {
    float aspect =
            static_cast<float>(viewport[2]) / static_cast<float>(viewport[3]);
    mat_projection_ =
            ndk_helper::Mat4::Perspective(aspect, 1.0f, CAM_NEAR, CAM_FAR);
  } else {
    float aspect =
            static_cast<float>(viewport[3]) / static_cast<float>(viewport[2]);
    mat_projection_ =
            ndk_helper::Mat4::Perspective(1.0f, aspect, CAM_NEAR, CAM_FAR);
  }
}

//--------------------------------------------------------------------------------
// Unload
//--------------------------------------------------------------------------------
void MoreTeapotsRenderer::Unload() {
  if (vbo_) {
    glDeleteBuffers(1, &vbo_);
    vbo_ = 0;
  }
  if (ubo_) {
    glDeleteBuffers(1, &ubo_);
    ubo_ = 0;
  }
  if (ibo_) {
    glDeleteBuffers(1, &ibo_);
    ibo_ = 0;
  }
  if (shader_param_.program_) {
    glDeleteProgram(shader_param_.program_);
    shader_param_.program_ = 0;
  }
}

//--------------------------------------------------------------------------------
// Update
//--------------------------------------------------------------------------------
void MoreTeapotsRenderer::Update(float fTime) {
  const float CAM_X = 0.f;
  const float CAM_Y = 0.f;
  const float CAM_Z = 2000.f;

  mat_view_ = ndk_helper::Mat4::LookAt(ndk_helper::Vec3(CAM_X, CAM_Y, CAM_Z),
                                       ndk_helper::Vec3(0.f, 0.f, 0.f),
                                       ndk_helper::Vec3(0.f, 1.f, 0.f));

  if (camera_) {
    camera_->Update();
    mat_view_ = camera_->GetTransformMatrix() * mat_view_ *
                camera_->GetRotationMatrix();
  }
}

//--------------------------------------------------------------------------------
// Render
//--------------------------------------------------------------------------------
void MoreTeapotsRenderer::Render() {
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mFramebuffer);
  glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mColorTexture, 0, 0);
  if (GL_FRAMEBUFFER_COMPLETE != glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER)) {
    std::terminate();
  }
  glClearColor(1, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);

  RenderReal();
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, mFramebuffer);
  glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

  //second pass
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mFramebuffer);
  glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mColorTexture, 0, 1);
  if (GL_FRAMEBUFFER_COMPLETE != glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER)) {
    std::terminate();
  }
  glClearColor(0, 1, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);

  RenderReal();

  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, mFramebuffer);
  glBlitFramebuffer(0, 0, width, height, 0, height+height, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

}

void MoreTeapotsRenderer::RenderReal() {
  glEnable(GL_DEPTH_TEST);
  // Bind the VBO
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);

  int32_t iStride = sizeof(TEAPOT_VERTEX);
  // Pass the vertex data
  glVertexAttribPointer(ATTRIB_VERTEX, 3, GL_FLOAT, GL_FALSE, iStride,
                        BUFFER_OFFSET(0));
  glEnableVertexAttribArray(ATTRIB_VERTEX);

  glVertexAttribPointer(ATTRIB_NORMAL, 3, GL_FLOAT, GL_FALSE, iStride,
                        BUFFER_OFFSET(3 * sizeof(GLfloat)));
  glEnableVertexAttribArray(ATTRIB_NORMAL);

  // Bind the IB
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);

  glUseProgram(shader_param_.program_);

  TEAPOT_MATERIALS material = {{1.0f, 1.0f, 1.0f, 10.f}, {0.1f, 0.1f, 0.1f}, };

  // Update uniforms
  //
  // using glUniform3fv here was troublesome..
  //
  glUniform4f(shader_param_.material_specular_, material.specular_color[0],
              material.specular_color[1], material.specular_color[2],
              material.specular_color[3]);
  glUniform3f(shader_param_.material_ambient_, material.ambient_color[0],
              material.ambient_color[1], material.ambient_color[2]);

  glUniform3f(shader_param_.light0_, 100.f, -200.f, -600.f);

  if (geometry_instancing_support_) {
    //
    // Geometry instancing, new feature in GLES3.0
    //

    // Update UBO
    glBindBuffer(GL_UNIFORM_BUFFER, ubo_);
    float* p = (float*)glMapBufferRange(
            GL_UNIFORM_BUFFER, 0, teapot_x_ * teapot_y_ * teapot_z_ *
                                  (ubo_matrix_stride_ * 2) * sizeof(float),
            GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT);
    float* mat_mvp = p;
    float* mat_mv = p + teapot_x_ * teapot_y_ * teapot_z_ * ubo_matrix_stride_;
    for (int32_t i = 0; i < teapot_x_ * teapot_y_ * teapot_z_; ++i) {
      // Rotation
      float x, y;
      vec_current_rotations_[i] += vec_rotations_[i];
      vec_current_rotations_[i].Value(x, y);
      ndk_helper::Mat4 mat_rotation =
              ndk_helper::Mat4::RotationX(x) * ndk_helper::Mat4::RotationY(y);

      // Feed Projection and Model View matrices to the shaders
      ndk_helper::Mat4 mat_v = mat_view_ * vec_mat_models_[i] * mat_rotation;
      ndk_helper::Mat4 mat_vp = mat_projection_ * mat_v;

      memcpy(mat_mvp, mat_vp.Ptr(), sizeof(mat_v));
      mat_mvp += ubo_matrix_stride_;

      memcpy(mat_mv, mat_v.Ptr(), sizeof(mat_v));
      mat_mv += ubo_matrix_stride_;
    }
    glUnmapBuffer(GL_UNIFORM_BUFFER);

    // Instanced rendering
    glDrawElementsInstanced(GL_TRIANGLES, num_indices_, GL_UNSIGNED_SHORT,
                            BUFFER_OFFSET(0),
                            teapot_x_ * teapot_y_ * teapot_z_);

  } else {
    // Regular rendering pass
    for (int32_t i = 0; i < teapot_x_ * teapot_y_ * teapot_z_; ++i) {
      // Set diffuse
      float x, y, z;
      vec_colors_[i].Value(x, y, z);
      glUniform4f(shader_param_.material_diffuse_, x, y, z, 1.f);

      // Rotation
      vec_current_rotations_[i] += vec_rotations_[i];
      vec_current_rotations_[i].Value(x, y);
      ndk_helper::Mat4 mat_rotation =
              ndk_helper::Mat4::RotationX(x) * ndk_helper::Mat4::RotationY(y);

      // Feed Projection and Model View matrices to the shaders
      ndk_helper::Mat4 mat_v = mat_view_ * vec_mat_models_[i] * mat_rotation;
      ndk_helper::Mat4 mat_vp = mat_projection_ * mat_v;
      glUniformMatrix4fv(shader_param_.matrix_projection_, 1, GL_FALSE,
                         mat_vp.Ptr());
      glUniformMatrix4fv(shader_param_.matrix_view_, 1, GL_FALSE, mat_v.Ptr());

      glDrawElements(GL_TRIANGLES, num_indices_, GL_UNSIGNED_SHORT,
                     BUFFER_OFFSET(0));
    }
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

//--------------------------------------------------------------------------------
// LoadShaders
//--------------------------------------------------------------------------------
bool MoreTeapotsRenderer::LoadShaders(SHADER_PARAMS* params, const char* strVsh,
                                      const char* strFsh) {
  //
  // Shader load for GLES2
  // In GLES2.0, shader attribute locations need to be explicitly specified
  // before linking
  //
  GLuint program;
  GLuint vertShader, fragShader;

  // Create shader program
  program = glCreateProgram();
  LOGI("Created Shader %d", program);

  // Create and compile vertex shader
  if (!ndk_helper::shader::CompileShader(&vertShader, GL_VERTEX_SHADER,
                                         strVsh)) {
    LOGI("Failed to compile vertex shader");
    glDeleteProgram(program);
    return false;
  }

  // Create and compile fragment shader
  if (!ndk_helper::shader::CompileShader(&fragShader, GL_FRAGMENT_SHADER,
                                         strFsh)) {
    LOGI("Failed to compile fragment shader");
    glDeleteProgram(program);
    return false;
  }

  // Attach vertex shader to program
  glAttachShader(program, vertShader);

  // Attach fragment shader to program
  glAttachShader(program, fragShader);

  // Bind attribute locations
  // this needs to be done prior to linking
  glBindAttribLocation(program, ATTRIB_VERTEX, "myVertex");
  glBindAttribLocation(program, ATTRIB_NORMAL, "myNormal");

  // Link program
  if (!ndk_helper::shader::LinkProgram(program)) {
    LOGI("Failed to link program: %d", program);

    if (vertShader) {
      glDeleteShader(vertShader);
      vertShader = 0;
    }
    if (fragShader) {
      glDeleteShader(fragShader);
      fragShader = 0;
    }
    if (program) {
      glDeleteProgram(program);
    }
    return false;
  }

  // Get uniform locations
  params->matrix_projection_ = glGetUniformLocation(program, "uPMatrix");
  params->matrix_view_ = glGetUniformLocation(program, "uMVMatrix");

  params->light0_ = glGetUniformLocation(program, "vLight0");
  params->material_diffuse_ = glGetUniformLocation(program, "vMaterialDiffuse");
  params->material_ambient_ = glGetUniformLocation(program, "vMaterialAmbient");
  params->material_specular_ =
      glGetUniformLocation(program, "vMaterialSpecular");

  // Release vertex and fragment shaders
  if (vertShader) glDeleteShader(vertShader);
  if (fragShader) glDeleteShader(fragShader);

  params->program_ = program;
  return true;
}

bool MoreTeapotsRenderer::LoadShadersES3(
    SHADER_PARAMS* params, const char* strVsh, const char* strFsh,
    std::map<std::string, std::string>& shaderParams) {
  //
  // Shader load for GLES3
  // In GLES3.0, shader attribute index can be described in a shader code
  // directly with layout() attribute
  //
  GLuint program;
  GLuint vertShader, fragShader;

  // Create shader program
  program = glCreateProgram();
  LOGI("Created Shader %d", program);

  // Create and compile vertex shader
  if (!ndk_helper::shader::CompileShader(&vertShader, GL_VERTEX_SHADER, strVsh,
                                         shaderParams)) {
    LOGI("Failed to compile vertex shader");
    glDeleteProgram(program);
    return false;
  }

  // Create and compile fragment shader
  if (!ndk_helper::shader::CompileShader(&fragShader, GL_FRAGMENT_SHADER,
                                         strFsh, shaderParams)) {
    LOGI("Failed to compile fragment shader");
    glDeleteProgram(program);
    return false;
  }

  // Attach vertex shader to program
  glAttachShader(program, vertShader);

  // Attach fragment shader to program
  glAttachShader(program, fragShader);

  // Link program
  if (!ndk_helper::shader::LinkProgram(program)) {
    LOGI("Failed to link program: %d", program);

    if (vertShader) {
      glDeleteShader(vertShader);
      vertShader = 0;
    }
    if (fragShader) {
      glDeleteShader(fragShader);
      fragShader = 0;
    }
    if (program) {
      glDeleteProgram(program);
    }

    return false;
  }

  // Get uniform locations
  params->light0_ = glGetUniformLocation(program, "vLight0");
  params->material_ambient_ = glGetUniformLocation(program, "vMaterialAmbient");
  params->material_specular_ =
      glGetUniformLocation(program, "vMaterialSpecular");

  // Release vertex and fragment shaders
  if (vertShader) glDeleteShader(vertShader);
  if (fragShader) glDeleteShader(fragShader);

  params->program_ = program;
  return true;
}

//--------------------------------------------------------------------------------
// Bind
//--------------------------------------------------------------------------------
bool MoreTeapotsRenderer::Bind(ndk_helper::TapCamera* camera) {
  camera_ = camera;
  return true;
}

//--------------------------------------------------------------------------------
// Helper functions
//--------------------------------------------------------------------------------
std::string MoreTeapotsRenderer::ToString(const int32_t i) {
  char str[64];
  snprintf(str, sizeof(str), "%d", i);
  return std::string(str);
}
