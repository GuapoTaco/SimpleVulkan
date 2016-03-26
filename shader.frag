#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable


in vec2 UV;

layout (binding = 1) uniform sampler2D tex;

layout (location = 0) out vec4 uFragColor;

void main() {
   uFragColor = textureLod(tex, UV, 0.0);
}
