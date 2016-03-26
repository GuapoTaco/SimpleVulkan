#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 vertUV;

layout (std140, binding = 0) uniform bufferVals {
        mat4 mvp;
} myBufferVals;

out vec2 UV;

void main() {
   gl_Position =  myBufferVals.mvp * vec4(pos, 1.f);
   UV = vertUV;
}
